export const sprites = {};
// Darkened copy used to dim tiles outside a selected city, built once per sprite.
export function darkImage(a) {
  if (!a.dark) {
    const c = document.createElement("canvas");
    c.width = a.image.width;
    c.height = a.image.height;
    const g = c.getContext("2d");
    g.drawImage(a.image, 0, 0);
    g.globalCompositeOperation = "source-atop";
    g.fillStyle = "rgba(8,24,40,0.42)";
    g.fillRect(0, 0, c.width, c.height);
    a.dark = c;
  }
  return a.dark;
}
// Cropped copy for DOM icons, so portraits fill their circles. Built lazily:
// encoding every sprite at startup would stall the first board animations.
export function spriteURL(name) {
  const a = sprites[name];
  if (!a) return `/assets/${name}.png`;
  if (!a.url) {
    const crop = document.createElement("canvas");
    crop.width = a.w;
    crop.height = a.h;
    crop.getContext("2d").drawImage(a.image, a.x, a.y, a.w, a.h, 0, 0, a.w, a.h);
    a.url = crop.toDataURL();
  }
  return a.url;
}
export async function loadAssets(redraw) {
  const response = await fetch("/assets/manifest.json");
  if (!response.ok) return;
  const manifest = await response.json();
  await Promise.all(
    Object.entries(manifest).map(
      ([key, entry]) =>
        new Promise((resolve) => {
          const image = new Image();
          image.onload = () => {
            const canvas = document.createElement("canvas");
            canvas.width = image.width;
            canvas.height = image.height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(image, 0, 0);
            const pixels = ctx.getImageData(
              0,
              0,
              image.width,
              image.height,
            ).data;
            let minX = image.width,
              minY = image.height,
              maxX = 0,
              maxY = 0;
            for (let y = 0; y < image.height; y++)
              for (let x = 0; x < image.width; x++)
                if (pixels[(y * image.width + x) * 4 + 3] > 20) {
                  minX = Math.min(minX, x);
                  maxX = Math.max(maxX, x);
                  minY = Math.min(minY, y);
                  maxY = Math.max(maxY, y);
                }
            sprites[key] = {
              image,
              x: minX,
              y: minY,
              w: maxX - minX + 1,
              h: maxY - minY + 1,
            };

            resolve();
          };
          image.onerror = resolve;
          image.src = "/assets/" + entry.file;
        }),
    ),
  );
  redraw();
  // Prepare the dimmed sprite copies while idle, so the first city tap is instant.
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 200));
  const pending = Object.values(sprites);
  const warm = () => {
    const t = performance.now();
    while (pending.length && performance.now() - t < 8) darkImage(pending.pop());
    if (pending.length) idle(warm);
  };
  idle(warm);
}
