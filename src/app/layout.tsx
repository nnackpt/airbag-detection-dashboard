"use client";

import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar, useSidebar } from "@/components/layout/Sidebar";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const { isCollapsed, toggle } = useSidebar();

  return (
    <html lang="en" className={`${inter.variable} h-full w-full`}>
      <body className={`${inter.className} antialiased`}>
        <Sidebar isCollapsed={isCollapsed} onToggleCollapse={toggle} />

        <main
          className="min-h-screen transition-all duration-300"
          style={{ marginLeft: "var(--sidebar-width, 320px)" }}
        >
          <div className="p-4 lg:p-8">{children}</div>
        </main>
      </body>
    </html>
  );
}
