import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import Landing from "./components/Landing";
import "./styles.css";

// Hash routing on purpose. The worker serves the built bundle from a static
// mount, so path routing would need a server-side SPA fallback; a hash needs
// nothing, survives a refresh, and is linkable.
//   #/      the landing page
//   #/room  the interview room
function Root() {
  const [route, setRoute] = useState(() => window.location.hash.replace(/^#/, "") || "/");

  useEffect(() => {
    const onHash = () => setRoute(window.location.hash.replace(/^#/, "") || "/");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("on-landing", route === "/");
    window.scrollTo(0, 0);
  }, [route]);

  if (route === "/") return <Landing onEnter={() => { window.location.hash = "#/room"; }} />;
  return <App />;
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
