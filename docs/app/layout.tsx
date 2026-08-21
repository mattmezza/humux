import "./global.css";
import { RootProvider } from "fumadocs-ui/provider/next";
import type { ReactNode } from "react";
import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-humux-display",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    template: "%s — humux Docs",
    default: "humux — Human Multiplexer",
  },
  description:
    "Documentation for humux — a squad of AI agents that acts like a full team of experts at your disposal, self-hosted in one container.",
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={spaceGrotesk.variable} suppressHydrationWarning>
      <head>
        <script defer src="https://analytics.casa.merola.co/script.js" data-website-id="388b4de6-a782-4d67-9cd3-d2a18df6bb60"></script>
      </head>
      <body>
        <RootProvider
          theme={{
            enabled: true,
            attribute: "class",
            defaultTheme: "system",
            enableSystem: true,
          }}
        >
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
