// Humux Admin — minimal service worker (#177)
// Provides the installability trigger for Lighthouse PWA audit
// and a basic offline fallback.

const CACHE = "humux-v1";
const SHELL = [
  "/static/manifest.json",
  "/static/icon-192.png",
  "/static/icon-512.png",
  "/static/style.css",
  "/offline",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL)),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(clients.claim());
});

self.addEventListener("fetch", (event) => {
  // Network-first for everything, fall back to cache for shell assets.
  event.respondWith(
    fetch(event.request)
      .then((resp) => {
        // Cache successful responses for shell assets.
        if (resp.ok && SHELL.includes(new URL(event.request.url).pathname)) {
          const clone = resp.clone();
          caches.open(CACHE).then((cache) => cache.put(event.request, clone));
        }
        return resp;
      })
      .catch(() => caches.match(event.request)),
  );
});
