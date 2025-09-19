"use client"

// import type { Metadata } from "next";
// import { Geist, Geist_Mono } from "next/font/google";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { useSidebar } from "@/components/layout/Sidebar";

// const geistSans = Geist({
//   variable: "--font-geist-sans",
//   subsets: ["latin"],
// });

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

// export const metadata: Metadata = {
//   title: "Airbag Detection Dashboard",
//   description: "Real-time monitoring and alerts",
// };

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const { isCollapsed, toggle } = useSidebar();

  return (
    <html lang="en" className="h-full w-full">
      <body className={`${inter.variable} antialiased`}>

        <Sidebar isCollapsed={isCollapsed} onToggleCollapse={toggle} />

        <main
          className="transition-all duration-300 min-h-screen"
          style={{
            marginLeft: 'var(--sidebar-width, 320px)'
          }}
        >
          <div className="p-4 lg:p-8">
            {children}
          </div>

        </main>
      </body>
    </html>
  );
}
