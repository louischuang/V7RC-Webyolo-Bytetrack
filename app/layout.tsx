import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VLA Testbed",
  description: "Chrome-local VLA testbed with YOLO, ByteTrack, Gemma4-E2B, and V7RC robot control.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
