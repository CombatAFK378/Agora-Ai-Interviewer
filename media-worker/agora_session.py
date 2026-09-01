"""The bot's presence inside an Agora channel: its ear and its mouth.

This is deliberately thin. It does two things:
  - EAR: receives the candidate's PCM and drops it on `inbound` (a thread-safe
    queue) for the pipeline to consume. The SDK fires this on its own native
    thread, so we do NO slow work here - just enqueue and return.
  - MOUTH: `speak()` accepts synthesized PCM for one agent turn, and a background
    thread paces it out to Agora at ~real time. `interrupt()` stops playback on
    barge-in and reports how much was already delivered.

One AgoraService exists per process (an SDK requirement). One AgoraSession =
one channel connection.
"""
import logging
import queue
import threading
import time
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

# Outbound audio is streamed in ~0.5 s chunks, paced at roughly real time so the
# SDK's buffer stays just barely ahead (no starvation gaps) while never getting
# more than one chunk ahead. That bound matters for interruption (ARCHITECTURE
# §4): when we stop feeding on barge-in, at most ~0.5 s of already-buffered audio
# can still play out. Phase 1 used one big 5 s slice for smoothness; that made
# barge-in feel broken (up to 5 s of the agent kept talking), so we trade a
# slightly finer push cadence for responsive interruption.
#
# If playback ever sounds choppy, raise PLAYBACK_CHUNK_SECS (at the cost of
# interruption latency) or nudge the pacing lead in _push_loop.
PLAYBACK_CHUNK_SECS = 0.5

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
    def __init__(self, session: "AgoraSession"):
        super().__init__()
        self._session = session

    def on_connected(self, conn, info, reason):
        logger.info(f"bot connected (uid={info.internal_uid})")

    def on_disconnected(self, conn, info, reason):
        logger.info("bot disconnected")

    def on_connecting(self, conn, info, reason):
        pass

    def on_user_joined(self, conn, user_id):
        logger.info(f"remote user joined: {user_id}")
        self._session._on_remote_join(user_id)

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
    def __init__(
        self,
        app_id: str,
        channel: str,
        bot_uid: int,
        token: str,
        sample_rate: int = 16000,
        on_user_joined=None,
    ):
        self.app_id = app_id
        self.channel = channel
        self.bot_uid = bot_uid
        self.token = token
        self.sample_rate = sample_rate
        # Called (once per join) when a remote user appears — the pipeline uses
        # it to deliver the scripted opener after the candidate has joined.
        self._on_user_joined = on_user_joined

        self.inbound: "queue.Queue[bytes]" = queue.Queue()
        # One continuous PCM buffer. Producers (streaming TTS) append audio in
        # whatever chunk sizes they arrive in; the push loop drains fixed ~0.5 s
        # slices, so playback pacing is independent of TTS chunk sizes.
        self._buffer = bytearray()
        self._out_lock = threading.Lock()
        self._stop = threading.Event()
        self._chunk_bytes = int(self.sample_rate * 2 * PLAYBACK_CHUNK_SECS)

        # Playback state for the current agent turn. Audio is streamed in:
        # begin_speech() opens the stream, add_speech() appends synthesized
        # chunks as they arrive (sentence by sentence), end_speech() closes it.
        # _stream_open keeps playback "live" through the gaps while later
        # sentences are still being synthesized.
        self._speaking = False
        self._stream_open = False
        self._total_bytes = 0        # bytes of PCM appended so far this turn
        self._play_start: Optional[float] = None  # monotonic time first chunk pushed

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
        self._conn_observer = _ConnObserver(self)
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

    def set_on_user_joined(self, callback):
        """Register the remote-join callback (set before start())."""
        self._on_user_joined = callback

    def _on_remote_join(self, user_id):
        if self._on_user_joined is not None:
            try:
                self._on_user_joined(user_id)
            except Exception:
                logger.exception("on_user_joined handler failed")

    def begin_speech(self):
        """Start a new streamed agent turn, replacing anything still playing."""
        with self._out_lock:
            self._buffer = bytearray()
            self._total_bytes = 0
            self._play_start = None
            self._speaking = True
            self._stream_open = True

    def add_speech(self, pcm_bytes: bytes):
        """Append a synthesized chunk to the current turn's playback buffer."""
        with self._out_lock:
            if not self._speaking:  # interrupted mid-stream — drop late audio
                return
            self._buffer.extend(pcm_bytes)
            self._total_bytes += len(pcm_bytes)

    def end_speech(self):
        """No more audio will be appended to this turn."""
        with self._out_lock:
            self._stream_open = False

    def interrupt(self) -> tuple[int, int]:
        """Stop playback immediately. Returns (delivered_bytes, synthesized_bytes).

        We can't un-push audio already handed to the SDK, so 'delivered' is a
        wall-clock estimate: playback is paced at real time, so the elapsed time
        since the first chunk went out maps to bytes heard (capped at the total).
        Slightly over-counts by the buffer depth (~<=0.5 s) — good enough to cite
        what the candidate actually heard for the truncation point.
        """
        with self._out_lock:
            self._buffer = bytearray()
            if self._play_start is None:
                delivered = 0
            else:
                elapsed = time.monotonic() - self._play_start
                delivered = min(self._total_bytes, max(0, int(elapsed * self.sample_rate * 2)))
            total = self._total_bytes
            self._speaking = False
            self._stream_open = False
            self._play_start = None
        return delivered, total

    def is_speaking(self) -> bool:
        return self._speaking

    def speaking_elapsed(self) -> float:
        """Seconds since the first audio of the current turn actually went out
        (0 if nothing has played yet). Used for the barge-in grace period."""
        with self._out_lock:
            if self._play_start is None:
                return 0.0
            return time.monotonic() - self._play_start

    def _push_loop(self):
        while not self._stop.is_set():
            chunk = None
            with self._out_lock:
                if self._speaking and self._buffer:
                    have = len(self._buffer)
                    # Take a full slice; take a partial tail only once the stream
                    # is closed (otherwise wait for more so we don't push slivers).
                    if have >= self._chunk_bytes:
                        take = self._chunk_bytes
                    elif not self._stream_open:
                        take = have
                    else:
                        take = 0
                    if take:
                        chunk = bytearray(self._buffer[:take])
                        del self._buffer[:take]
                        if self._play_start is None:
                            self._play_start = time.monotonic()

            if chunk is not None:
                try:
                    self._conn.push_audio_pcm_data(chunk, self.sample_rate, 1)
                except Exception:
                    logger.exception("push_audio_pcm_data failed")
                # Pace just under real time (by the slice's actual duration) so the
                # SDK buffer stays ~one chunk ahead: gap-free, but never so far
                # ahead that interruption lags.
                time.sleep((len(chunk) / (self.sample_rate * 2)) * 0.9)
            else:
                self._maybe_finish_playback()
                time.sleep(0.02)

    def _maybe_finish_playback(self):
        """Mark playback done once the last pushed audio has had time to play."""
        with self._out_lock:
            # While the stream is open, an empty buffer just means the next
            # sentence is still being synthesized — stay "speaking".
            if not self._speaking or self._buffer or self._stream_open:
                return
            if self._play_start is None:
                # Nothing was ever pushed (e.g. empty turn) — just finish.
                self._speaking = False
                return
            played = time.monotonic() - self._play_start
            if played >= self._total_bytes / (self.sample_rate * 2) + 0.3:
                self._speaking = False
                self._play_start = None

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
