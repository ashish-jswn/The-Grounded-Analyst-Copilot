/**
 * Deliberately minimal.
 *
 * THE BACKEND PROXY IS **NOT** HERE. It used to be a `rewrites()` entry, and
 * that was wrong: `next build` evaluates `rewrites()` once and freezes the
 * result into `.next/routes-manifest.json`, so BACKEND_URL set at `next start`
 * was silently ignored and every call went to the build-time default. The proxy
 * now lives in `app/api/[...path]/route.ts`, which reads the environment per
 * request — see that file.
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
