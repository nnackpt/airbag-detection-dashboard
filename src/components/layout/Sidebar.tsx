"use client"

import Link from "next/link";
import React, { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";

import {
  ChartBarIcon,
  ArrowUpIcon,
  HomeIcon
} from "@heroicons/react/24/outline";

import { motion, AnimatePresence, MotionGlobalConfig } from "framer-motion";
import Image from "next/image";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";

// Types
interface MenuItem {
  label: string;
  href: string;
  icon?: React.ComponentType<{ className?: string }>;
}

interface SidebarMenuGroup {
  key: string;
  label: string;
  items: MenuItem[];
  icon?: React.ComponentType<{ className?: string }>;
}

interface SidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

// Sidebar Component
export const Sidebar: React.FC<SidebarProps> = ({ isCollapsed, onToggleCollapse }) => {
  const pathname = usePathname();
  const [expandedMenus, setExpandedMenus] = React.useState<{[key: string]: boolean}>({});
  // Changed showSettings to isSettingsExpanded for better clarity as it's now an inline expansion
  const [isSettingsExpanded, setIsSettingsExpanded] = React.useState(false);
  const [showScrollToTop, setShowScrollToTop] = React.useState(false);

  const [fontSizeOpen, setFontSizeOpen] = useState(false);
  const fontSizeRef = useRef<HTMLDivElement>(null);
  const lastScrollY = useRef(0)

  // User info state
  const [user, setUser] = React.useState<{ userName: string } | null>(null);
  const [loading, setLoading] = React.useState(true);

  const [fontSize, setFontSize] = React.useState<string>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("fontSize") || "base";
    }
    return "base";
  });

  const [animationsEnabled, setAnimationsEnabled] = React.useState<boolean>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("animationsEnabled") !== "false";
    }
    return true;
  });

  const COLLAPSED_WIDTH = 80
  const [expandedWidth, setExpandedWidth] = React.useState<number>(320)

  const getExpandedWidth = (vw: number) => {
    if (vw >= 1600) return 320
    if (vw >= 1440) return 300
    if (vw >= 1280) return 280
    if (vw >= 1024) return 260
    return 240
  }

  React.useEffect(() => {
    const apply = () => {
      const ew = getExpandedWidth(window.innerWidth)
      setExpandedWidth(ew)
      const real = (isCollapsed ? COLLAPSED_WIDTH : ew)
      document.documentElement.style.setProperty("--sidebar-width", `${real}px`)
    }

    apply()
    window.addEventListener("resize", apply, { passive: true })
    return () => window.removeEventListener("resize", apply)
  }, [isCollapsed])

  // Scroll to top functionality
  React.useEffect(() => {
    const handleScroll = () => {
      const y = window.pageYOffset || document.documentElement.scrollTop
      const prev = lastScrollY.current
      const scrollingDown = y > prev

      if (y <= 300) {
        setShowScrollToTop(false)
      } else if (scrollingDown) { 
        setShowScrollToTop(true) // >300 Show BTN
      } else {
        setShowScrollToTop(false)
      }

      lastScrollY.current = y
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  // Fetch user info
  React.useEffect(() => {
    const fetchUser = async () => {
      try {
        const res = await fetch("https://alvs-thappdev01:44324/api/UserInfo/current", {
          credentials: "include",
        });
        if (res.ok) {
          const data = await res.json();
          setUser(data);
        } else {
          setUser(null);
        }
      } catch {
        setUser(null);
      } finally {
        setLoading(false);
      }
    };
    fetchUser();
  }, []);

  // เพิ่ม useEffect สำหรับ Font Size dropdown
  useEffect(() => {
    const handleClickOutside = (_e: MouseEvent) => {
      if (fontSizeRef.current && !fontSizeRef.current.contains(_e.target as Node)) {
        setFontSizeOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // เพิ่มข้อมูล options สำหรับ Font Size
  const fontSizeOptions = [
    { value: "small", label: "Small" },
    { value: "base", label: "Medium" },
    { value: "large", label: "Large" }
  ];

  // Font size effect
  React.useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('text-sm', 'text-base', 'text-lg');

    if (fontSize === 'small') root.classList.add('text-sm');
    else if (fontSize === 'large') root.classList.add('text-lg');
    else root.classList.add('text-base');
    localStorage.setItem("fontSize", fontSize);
  }, [fontSize]);

  // Animations effect
  React.useEffect(() => {
    MotionGlobalConfig.skipAnimations = !animationsEnabled;
    localStorage.setItem("animationsEnabled", String(animationsEnabled));
  }, [animationsEnabled]);

  // Switch Color
  const availableColors = {
    'blue': '#005496', // corporate blue (default)
    'gray': '#4e5f6e', // neutral slate
    'indigo': '#4f46e5', // popular for dashboards
    'teal': '#0d9488', // clean & modern
    'amber': '#f59e0b' // warm accent
  }

  const [primaryColor, setPrimaryColor] = React.useState<string>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("primaryColor") || "#005496"
    }
    return "#005496" // default
  })

  
  React.useEffect(() => {
    const darkShades: { [key: string]: string } = {
      '#005496': '#003d73',
      '#4e5f6e': '#1d252b',
      '#4f46e5': '#4338ca',
      '#0d9488': '#0f766e',
      '#f59e0b': '#b45309'
    };
  
    const lightShades: { [key: string]: string } = {
      '#005496': '#009EE3', 
      '#4e5f6e': '#a3b1bc',
      '#4f46e5': '#818cf8',
      '#0d9488': '#14b8a6',
      '#f59e0b': '#fbbf24'
    };
    
    const cleanColor = primaryColor.trim().replace(/^##/, '#');
    const darkColor = darkShades[cleanColor] || '#003d73';
    const lightColor = lightShades[cleanColor] || '#b3d9ff';

    document.documentElement.style.setProperty('--primary-color', cleanColor);
    document.documentElement.style.setProperty('--primary-color-dark', darkColor);
    document.documentElement.style.setProperty('--primary-color-light', lightColor);

    localStorage.setItem("primaryColor", cleanColor);
  }, [primaryColor])

  // Menu data with icons
  const menuData: SidebarMenuGroup[] = [
    {
      key: 'home',
      label: 'Home',
      icon: HomeIcon,
      items: [
        { label: 'Dashboard', href: '/', icon: HomeIcon }
      ]
    },
    {
      key: 'reports',
      label: 'Reports',
      icon: ChartBarIcon,
      items: [
        { label: 'Reports', href: '/reports', icon: ChartBarIcon }
      ]
    }
  ]

  const toggleMenu = (key: string) => {
    if (isCollapsed) {
      onToggleCollapse();
      setTimeout(() => {
        setExpandedMenus({ [key]: true });
      }, 300);
    } else {
      setExpandedMenus(prev => ({
        ...prev,
        [key]: !prev[key]
      }));
    }
  };

  const isMenuActive = (items: MenuItem[]) => {
    return items.some(item => pathname === item.href);
  };

  return (
    <>
      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: isCollapsed ? COLLAPSED_WIDTH : expandedWidth }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="fixed left-0 top-0 h-full shadow-2xl z-40 flex flex-col border-r border-white/10"
        style={{
          background: '#164799',
        }}
      >
        {/* Header */}
        <div className="p-4 border-b border-white/10">
          <div className="flex items-center justify-between">
            {/* Wrapped logo and title in an anchor tag for homepage navigation */}
            <AnimatePresence>
              {!isCollapsed && (
                <motion.a
                  href="/"
                  className="flex items-center space-x-3 hover:scale-110"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.2 }}
                >
                  <Image
                    src="/autoliv_logo.png"
                    alt="Autoliv"
                    width={60}
                    height={24}
                    className="w-15 h-6 flex-shrink-0"
                  />
                  {/* <div>
                    <h2 className="text-white font-bold text-lg whitespace-nowrap">RBAC System</h2>
                  </div> */}
                </motion.a>
              )}
            </AnimatePresence>

            <div className="relative">
              <button
                onClick={onToggleCollapse}
                aria-label={isCollapsed ? "Open sidebar" : "Close sidebar"}
                className={`
                  group inline-flex items-center justify-center cursor-pointer
                  ${isCollapsed ? "h-10 w-10" : "h-9 w-9"}
                  rounded-md text-white/80 hover:text-white hover:bg-white/10
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40
                `}
              >
                <span
                  className={`
                    inline-flex items-center justify-center
                    ${isCollapsed ? "translate-x-[2px]" : ""}  
                  `}
                >
                  {isCollapsed ? (
                    <>
                      <Image 
                        src="/autoliv_logo.png"
                        alt="logo"
                        width={60}
                        height={24}
                        className="h-5 w-10 opacity-100 group-hover:opacity-0 transition-opacity duration-150"
                        priority
                      />
                      <PanelLeftOpen className="h-6 w-6 absolute opacity-0 group-hover:opacity-100 transition-opacity duration-150" />
                    </>
                  ) : (
                    <PanelLeftClose className="h-5 w-5" />
                  )}
                </span>

                {/* Tooltip */}
                <div
                  role="tooltip"
                  className={`
                    pointer-events-none absolute z-50 whitespace-nowrap rounded-md
                    bg-gray-900/95 text-white text-[11px] font-medium px-2 py-1 shadow-lg
                    opacity-0 transition-all duration-150
                    group-hover:opacity-100 group-focus-visible:opacity-100
                    ${isCollapsed
                      ? "left-full top-1/2 -translate-y-1/2 ml-2"
                      : "left-1/2 top-full -translate-x-1/2 mt-2"}
                  `}
                >
                  {isCollapsed ? "Open sidebar" : "Close sidebar"}
                  <span 
                    className={`
                      absolute w-2 h-2 rotate-45 bg-gray-900/95
                      ${isCollapsed
                        ? "left-0 top-1/2 -translate-y-1/2 -ml-1"
                        : "-top-1 left-1/2 -translate-x-1/2"}
                      `}
                    />
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Title */}
        {/* <AnimatePresence>
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="px-4 py-2 border-b border-white/10 overflow-hidden"
            >
              <h1 className="text-white/90 font-semibold text-sm uppercase tracking-wider text-center">
                Application Access Control
              </h1>
            </motion.div>
          )}
        </AnimatePresence> */}

        {/* User Info */}
        {/* <div className="p-4 border-b border-white/10">
          <div className="flex items-center space-x-3 bg-white/10 backdrop-blur-sm rounded-lg px-1.5 py-2">
            <div className="p-2 bg-white/20 rounded-full flex-shrink-0">
              <UserIcon className="w-5 h-5 text-white" />
            </div>
            <AnimatePresence>
              {!isCollapsed && (
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.2 }}
                  className="flex-1 min-w-0"
                >
                  <p className="text-white font-medium text-sm truncate">
                    {loading ? "Loading..." : user?.userName || "User"}
                  </p>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-white/70 text-xs">Online</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div> */}


        {/* Navigation */}
        <div className="flex-1 overflow-y-auto no-scrollbar p-4">
          <nav className="space-y-2">
            {menuData.map((menu) => {
              const isExpanded = expandedMenus[menu.key];
              const isActive = isMenuActive(menu.items);
              const Icon = menu.icon;

              return (
                <div key={menu.key} className="space-y-1">
                  <Link
                    href={menu.items[0].href} // ใช้ href ของ item แรก
                    className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg transition-all duration-200 group relative cursor-pointer ${
                      isActive
                        ? 'bg-white/20 text-white shadow-lg'
                        : 'text-white/80 hover:text-white hover:bg-white/10'
                    } ${isCollapsed ? 'justify-center' : ''}`}
                    title={isCollapsed ? menu.label : undefined}
                  >
                    {/* Icon remains in fixed position */}
                    {Icon && <Icon className="w-5 h-5 flex-shrink-0 absolute left-3 top-1/2 transform -translate-y-1/2" />}
                    
                    {/* Label and space for icon */}
                    <div className="flex items-center w-full">
                      <div className="w-5 h-5 flex-shrink-0 mr-3"></div>
                      <AnimatePresence>
                        {!isCollapsed && (
                          <motion.span
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            transition={{ duration: 0.2 }}
                            className="font-medium text-sm whitespace-nowrap"
                          >
                            {menu.label}
                          </motion.span>
                        )}
                      </AnimatePresence>
                    </div>

                    {/* <AnimatePresence>
                      {!isCollapsed && (
                        <motion.div
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 20 }}
                          transition={{ duration: 0.2 }}
                        >
                          <ChevronRightIcon
                            className={`w-4 h-4 transition-transform duration-200 ${
                              isExpanded ? 'rotate-90' : ''
                            }`}
                          />
                        </motion.div>
                      )}
                    </AnimatePresence> */}
                  </Link>

                  <AnimatePresence>
                    {isExpanded && !isCollapsed && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                      >
                        <div className="pl-8 space-y-1">
                          {menu.items.map((item, index) => {
                            const isItemActive = pathname === item.href;
                            return (
                              <Link
                                key={index}
                                href={item.href}
                                className={`block px-3 py-2 rounded-lg text-sm transition-all duration-200 ${
                                  isItemActive
                                    ? 'bg-[var(--primary-color)] text-white shadow-md'
                                    : 'text-white/70 hover:text-white hover:bg-[var(--primary-color-dark)]'
                                }`}
                              >
                                <div className="flex items-center space-x-2">
                                  <div className={`w-1.5 h-1.5 rounded-full ${
                                    isItemActive ? 'bg-white' : 'bg-white/40'
                                  }`} />
                                  <span>{item.label}</span>
                                </div>
                              </Link>
                            );
                          })}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
          </nav>

          <AnimatePresence>
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ duration: 0.3, delay: 0.1 }}
                className="mt-4 pt-4 border-t border-white/10"
              >
                {/* SPEC Title */}
                <div className="px-3 mb-3">
                  <h3 className="text-white font-bold text-xs uppercase tracking-wider">
                    SPEC
                  </h3>
                </div>

                {/* Compact Spec Display */}
                <div className="px-3 space-y-4">
                  {/* AMBIENT */}
                  <div>
                    <div className="text-white/90 font-semibold mb-2 uppercase text-xs tracking-wide">
                      AMBIENT
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {[
                        { target: "F#1", spec: "≤17" },
                        { target: "F#2", spec: "≤20" },
                        { target: "R#3", spec: "≤19" }
                      ].map((item, idx) => (
                        <div
                          key={idx}
                          className="relative rounded-lg px-3 py-2 text-center ring-1 ring-white/20 shadow-[0_10px_18px_rgba(0,0,0,.35)] before:content-[''] before:absolute before:inset-x-0 before:top-0 before:h-1/2 before:rounded-t-lg before:bg-gradient-to-b before:from-white/30 before:to-transparent after:content-[''] after:absolute after:inset-x-0 after:bottom-0 after:h-1/3 after:rounded-b-lg after:bg-gradient-to-t after:from-black/20 after:to-transparent"
                          style={{ backgroundColor: "#39c06b" }}
                        >
                          <div className="text-white font-semibold text-sm">{item.target}</div>
                          <div className="text-white/90 text-xs">{item.spec}ms</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* HOT */}
                  <div>
                    <div className="text-white/90 font-semibold mb-2 uppercase text-xs tracking-wide">
                      HOT
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {[
                        { target: "F#1", spec: "≤17" },
                        { target: "F#2", spec: "≤20" },
                        { target: "R#3", spec: "≤19" }
                      ].map((item, idx) => (
                        <div
                          key={idx}
                          className="relative rounded-lg px-3 py-2 text-center ring-1 ring-white/20 shadow-[0_10px_18px_rgba(0,0,0,.35)] before:content-[''] before:absolute before:inset-x-0 before:top-0 before:h-1/2 before:rounded-t-lg before:bg-gradient-to-b before:from-white/30 before:to-transparent after:content-[''] after:absolute after:inset-x-0 after:bottom-0 after:h-1/3 after:rounded-b-lg after:bg-gradient-to-t after:from-black/20 after:to-transparent"
                          style={{ backgroundColor: "#e23b2e" }}
                        >
                          <div className="text-white font-semibold text-sm">{item.target}</div>
                          <div className="text-white/90 text-xs">{item.spec}ms</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* COLD */}
                  <div>
                    <div className="text-white/90 font-semibold mb-2 uppercase text-xs tracking-wide">
                      COLD
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {[
                        { target: "F#1", spec: "≤21" },
                        { target: "F#2", spec: "≤22" },
                        { target: "R#3", spec: "≤20" }
                      ].map((item, idx) => (
                        <div
                          key={idx}
                          className="relative rounded-lg px-3 py-2 text-center ring-1 ring-white/20 shadow-[0_10px_18px_rgba(0,0,0,.35)] before:content-[''] before:absolute before:inset-x-0 before:top-0 before:h-1/2 before:rounded-t-lg before:bg-gradient-to-b before:from-white/30 before:to-transparent after:content-[''] after:absolute after:inset-x-0 after:bottom-0 after:h-1/3 after:rounded-b-lg after:bg-gradient-to-t after:from-black/20 after:to-transparent"
                          style={{ backgroundColor: "#3fa0f5" }}
                        >
                          <div className="text-white font-semibold text-sm">{item.target}</div>
                          <div className="text-white/90 text-xs">{item.spec}ms</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        
      </motion.aside>

      {/* Scroll to Top Button */}
      <AnimatePresence>
        {showScrollToTop && (
          <motion.button
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3 }}
            onClick={scrollToTop}
            // --- Positioning and Styling ---
            className="fixed bottom-6 left-1/2 -translate-x-1/2 z-30
                      bg-white px-2 py-1.5 rounded-full shadow-lg hover:shadow-xl
                      transition-all duration-300 hover:scale-105 backdrop-blur-sm border border-white/20
                      flex items-center space-x-2 font-semibold cursor-pointer"
            aria-label="Back to top"
          >
            <ArrowUpIcon className="w-4 h-4" style={{ color: 'var(--primary-color)' }} />
            <span className="text-sm" style={{ color: 'var(--primary-color)' }}>Back To Top</span>
          </motion.button>
        )}
      </AnimatePresence>
    </>
  );
};

// Hook for managing sidebar state
export const useSidebar = () => {
  const [isCollapsed, setIsCollapsed] = React.useState(false);

  const toggle = () => setIsCollapsed(!isCollapsed);
  const collapse = () => setIsCollapsed(true);
  const expand = () => setIsCollapsed(false);

  return { isCollapsed, toggle, collapse, expand };
};