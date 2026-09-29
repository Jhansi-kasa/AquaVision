import React from 'react';
import { NAV } from '../../data.js';
import icon from '../../assets/icon.jpeg';

export const Sidebar = ({
  activePage,
  onNavigate,
  mobileOpen,
  onCloseMobile,
}) => {
  return (
    <>
      {/* =========================================================
          MOBILE BACKDROP
          ========================================================= */}
      {mobileOpen && (
        <div
          className="sidebar-backdrop"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}

      {/* =========================================================
          SIDEBAR
          
          Desktop:
          - Fixed to viewport
          - Always occupies full viewport height
          - Does NOT move when page scrolls
          
          Mobile:
          - Fixed drawer
          - Slides in/out
          ========================================================= */}
      <aside
        className={`sidebar${mobileOpen ? ' mobile-open' : ''}`}
        aria-label="Main navigation"
      >
        {/* =======================================================
            BRAND
            ======================================================= */}
        <div className="sidebar-brand">
          {/* Aqua Vision icon */}
          <div className="sidebar-brand-icon">
            <img
              src={icon}
              alt="Aqua Vision"
            />
          </div>

          {/* Brand text */}
          <div className="sidebar-brand-text">
            <div className="sidebar-brand-title">
              Aqua Vision
            </div>

            <div className="sidebar-brand-subtitle">
              UNDERWATER OPS
            </div>
          </div>

          {/* Mobile close button */}
          <button
            type="button"
            onClick={onCloseMobile}
            className="sidebar-close lg:hidden"
            aria-label="Close menu"
          >
            ×
          </button>
        </div>

        {/* =======================================================
            NAVIGATION
            ======================================================= */}
        <nav className="sidebar-nav">
          {NAV.map((item, index) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;

            return (
              <button
                key={item.id ?? `nav-item-${index}`}
                type="button"
                onClick={() => {
                  onNavigate(item.id);

                  if (onCloseMobile) {
                    onCloseMobile();
                  }
                }}
                className={`sidebar-item${
                  isActive ? ' active' : ''
                }`}
              >
                <span className="sidebar-item-icon">
                  <Icon />
                </span>

                <span className="sidebar-item-label">
                  {item.label}
                </span>
              </button>
            );
          })}
        </nav>

        {/* =======================================================
            SIDEBAR FOOTER / STATUS
            ======================================================= */}
        <div className="sidebar-status">
          <span className="sidebar-status-dot" />

          <span className="sidebar-status-text">
            YOLOv11n model online
          </span>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;