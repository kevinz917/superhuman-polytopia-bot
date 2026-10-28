import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./styles/base.css";
import "./styles/mobile.css";
import "./styles/app.css";

// No StrictMode: the WebGL board must mount exactly once.
createRoot(document.getElementById("root")!).render(<App />);
