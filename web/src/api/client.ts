// Thin client for the loopback studio API; engine work is polled as jobs.

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Studio-Client": "1" },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok) throw Error(result.error || response.statusText);
  return result as T;
}

interface Job<T> {
  job_id?: string;
  status?: "queued" | "running" | "done" | "error";
  result?: T;
  error?: string;
}

export async function waitJob<T>(started: Job<T> | T): Promise<T> {
  const id = (started as Job<T>).job_id;
  if (!id) return started as T;
  for (;;) {
    await new Promise((r) => setTimeout(r, 200));
    const job = await api<Job<T>>("/api/jobs/" + id);
    if (job.status === "done") return job.result as T;
    if (job.status === "error") throw Error(job.error);
  }
}

export const job = async <T>(path: string, body?: unknown) =>
  waitJob<T>(await api<Job<T> | T>(path, body));
