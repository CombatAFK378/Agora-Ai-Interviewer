"""The bot's presence inside an Agora channel: its ear and its mouth.

This is deliberately thin. It does two things:
  - EAR: receives the candidate's PCM and drops it on `inbound` (a thread-safe
    queue) for the pipeline to consume. The SDK fires this on its own native
    thread, so we do NO slow work here - just enqueue and return.
  - MOUTH: `enqueue_playback()` accepts synthesized PCM, and a background thread
    paces it out to Agora at real-time using the SDK's flow-control check.

One AgoraService exists per process (an SDK requirement). One AgoraSession =
one channel connection.
"""
import logging
import queue
import threading
import time
from collections import deque
from typing import Optional

from agora.rtc.agora_service import (
    AgoraService,
    AgoraServiceConfig,
    AudioSubscriptionOptions,
    RTCConnConfig,
)
from agora.rtc.audio_frame_observer import IAudioFrameObserver
from agora.rtc.local_user_observer import IRTCLocalUserObserver
from agora.rtc.rtc_connection_observer import IRTCConnectionObserver
from agora.rtc.agora_base import (
    AudioProfileType,
    AudioPublishType,
    AudioScenarioType,
    ChannelProfileType,
    ClientRoleType,
    RtcConnectionPublishConfig,
    SenderOptions,
    TCcMode,
    VideoCodecType,
    VideoPublishType,
)

logger = logging.getLogger(__name__)

# Outbound audio is pushed in large slices so the SDK plays it back seamlessly.
# Small slices leave audible gaps between pushes (each push waits for the
# previous to fully drain). Agora's own examples push up to ~5 s at once, so we
# use a 5 s slice: short replies become a single gap-free push.
PLAYBACK_CHUNK_BYTES = 160000  # 80000 samples * 2 bytes = 5 s @ 16 kHz

_service: Optional[AgoraService] = None
_service_lock = threading.Lock()


def get_agora_service(app_id: str) -> AgoraService:
    """Lazily create the one AgoraService this process is allowed to have."""
    global _service
    with _service_lock:
        if _service is None:
            cfg = AgoraServiceConfig()
            cfg.appid = app_id
            cfg.log_path = "./agora_rtc_log/agorasdk.log"
            svc = AgoraService()
            svc.initialize(cfg)
            _service = svc
            logger.info("AgoraService initialized")
    return _service


class _ConnObserver(IRTCConnectionObserver):
    def on_connected(self, conn, info, reason):
        logger.info(f"bot connected (uid={info.internal_uid})")

    def on_disconnected(self, conn, info, reason):
        logger.info("bot disconnected")

    def on_connecting(self, conn, info, reason):
        pass

    def on_user_joined(self, conn, user_id):
        logger.info(f"remote user joined: {user_id}")

    def on_aiqos_capability_missing(self, conn, recommend_audio_scenario):
        return recommend_audio_scenario


class _UserObserver(IRTCLocalUserObserver):
    """No-op; required so the SDK has a local-user observer registered."""
    pass


class _InboundAudioObserver(IAudioFrameObserver):
    def __init__(self, on_pcm):
        super().__init__()
        self._on_pcm = on_pcm

    def on_playback_audio_frame_before_mixing(
        self, local_user, channel_id, uid, frame, vad_result_state, vad_result_bytearray
    ):
        try:
            self._on_pcm(bytes(frame.buffer))
        except Exception:  # never let a listener error kill the audio callback
            logger.exception("inbound audio handler failed")
        return 0

    def on_record_audio_frame(self, *args):
        return 0

    def on_playback_audio_frame(self, *args):
        return 0

    def on_ear_monitoring_audio_frame(self, *args):
        return 0

    def on_get_audio_frame_position(self, *args):
        return 0


class AgoraSession:
    def __init__(self, app_id: str, channel: str, bot_uid: int, token: str, sample_rate: int = 16000):
        self.app_id = app_id
        self.channel = channel
        self.bot_uid = bot_uid
        self.token = token
        self.sample_rate = sample_rate

        self.inbound: "queue.Queue[bytes]" = queue.Queue()
        self._outbound: deque[bytearray] = deque()
        self._out_lock = threading.Lock()
        self._stop = threading.Event()

        self._conn = None
        self._push_thread: Optional[threading.Thread] = None
        # Keep observer refs alive; the SDK holds only weak/native references.
        self._conn_observer = None
        self._user_observer = None
        self._audio_observer = None

    def start(self):
        svc = get_agora_service(self.app_id)

        sub_opt = AudioSubscriptionOptions(
            packet_only=0,
            pcm_data_only=1,
            bytes_per_sample=2,
            number_of_channels=1,
            sample_rate_hz=self.sample_rate,
        )
        conn_cfg = RTCConnConfig(
            auto_subscribe_audio=1,
            auto_subscribe_video=0,
            client_role_type=ClientRoleType.CLIENT_ROLE_BROADCASTER,
            channel_profile=ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
            audio_recv_media_packet=0,
            audio_subs_options=sub_opt,
            enable_audio_recording_or_playout=0,
        )
        pub_cfg = RtcConnectionPublishConfig(
            audio_profile=AudioProfileType.AUDIO_PROFILE_DEFAULT,
            audio_scenario=AudioScenarioType.AUDIO_SCENARIO_AI_SERVER,
            is_publish_audio=True,
            is_publish_video=False,
            audio_publish_type=AudioPublishType.AUDIO_PUBLISH_TYPE_PCM,
            video_publish_type=VideoPublishType.VIDEO_PUBLISH_TYPE_NONE,
            video_encoded_image_sender_options=SenderOptions(
                target_bitrate=4160,
                cc_mode=TCcMode.CC_ENABLED,
                codec_type=VideoCodecType.VIDEO_CODEC_H264,
            ),
        )

        self._conn = svc.create_rtc_connection(conn_cfg, pub_cfg)
        self._conn_observer = _ConnObserver()
        self._conn.register_observer(self._conn_observer)
        self._conn.connect(self.token, self.channel, str(self.bot_uid))

        local_user = self._conn.get_local_user()
        self._user_observer = _UserObserver()
        self._conn.register_local_user_observer(self._user_observer)

        # Must be set before registering the audio observer.
        local_user.set_playback_audio_frame_before_mixing_parameters(1, self.sample_rate)
        self._audio_observer = _InboundAudioObserver(self._on_inbound_pcm)
        self._conn.register_audio_frame_observer(self._audio_observer, 0, None)

        self._conn.publish_audio()

        self._stop.clear()
        self._push_thread = threading.Thread(target=self._push_loop, daemon=True, name="agora-push")
        self._push_thread.start()
        logger.info(f"AgoraSession started on channel={self.channel} bot_uid={self.bot_uid}")

    def _on_inbound_pcm(self, pcm: bytes):
        self.inbound.put_nowait(pcm)

    def enqueue_playback(self, pcm_bytes: bytes):
        """Queue synthesized speech to be spoken into the channel.

        Chunks are stored as bytearray (mutable): the SDK pushes them via
        ctypes.from_buffer(), which rejects immutable bytes.
        """
        with self._out_lock:
            for i in range(0, len(pcm_bytes), PLAYBACK_CHUNK_BYTES):
                self._outbound.append(bytearray(pcm_bytes[i : i + PLAYBACK_CHUNK_BYTES]))

    def _push_loop(self):
        while not self._stop.is_set():
            chunk = None
            if self._conn is not None and self._conn.is_push_to_rtc_completed():
                with self._out_lock:
                    if self._outbound:
                        chunk = self._outbound.popleft()
            if chunk is not None:
                try:
                    self._conn.push_audio_pcm_data(chunk, self.sample_rate, 1)
                except Exception:
                    logger.exception("push_audio_pcm_data failed")
                time.sleep(0.02)
            else:
                time.sleep(0.04)

    def stop(self):
        self._stop.set()
        if self._push_thread is not None:
            self._push_thread.join(timeout=2)
        if self._conn is not None:
            try:
                self._conn.disconnect()
            except Exception:
                pass
            try:
                self._conn.release()
            except Exception:
                pass
        self._conn = None
        logger.info("AgoraSession stopped")
