import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "glassbox trace viewer",
  description: "Interactive playback for transformer internals"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
