import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "V7RC WebYOLO ByteTrack",
  description: "Chrome-local YOLO, ByteTrack, and Gemma4-E2B vision chat.",
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
