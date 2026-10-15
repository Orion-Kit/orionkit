// Minimal service worker so Orion installs as an app. Always network-first: Orion is live, nothing is cached for offline use.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", () => {});
