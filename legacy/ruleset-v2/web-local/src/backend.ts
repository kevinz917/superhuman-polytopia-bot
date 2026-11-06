// Main-thread side of the in-browser backend: forwards Studio API calls to local/worker.ts.
let worker: Worker | null = null;
let next = 0;
const calls = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

function start() {
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e: MessageEvent<{ id: number; value?: unknown; error?: string }>) => {
    const call = calls.get(e.data.id);
    calls.delete(e.data.id);
    if (e.data.error !== undefined) call?.reject(Error(e.data.error));
    else call?.resolve(e.data.value);
  };
  worker.onerror = (e) => {
    for (const call of calls.values()) call.reject(Error(e.message || "The in-browser engine failed to start"));
    calls.clear();
  };
  return worker;
}

export function localApi<T>(path: string, body?: unknown): Promise<T> {
  const w = worker ?? start();
  const id = next++;
  return new Promise<T>((resolve, reject) => {
    calls.set(id, { resolve: resolve as (v: unknown) => void, reject });
    w.postMessage({ id, path, body });
  });
}
