import React from 'react';

const PAGE_HEADINGS = {
  dashboard: {
    title: 'Dashboard Overview',
    
  },

  surveys: {
    title: 'Survey Management',
    
  },

  sonar: {
    title: 'Sonar Analysis & Detection',
    
  },

  review: {
    title:'Detection Review',
    
  },

  map: {
    title: 'Risk Map',
   
  },

  cleanup: {
    
    title: 'Cleanup Route Planning',
    
  },

  verify: {
    title: 'Before / After Cleanup Verification',
    
  },

  reports: {
    title: 'Survey Reports',
    
  },

  settings: {
    title: 'Model & Risk Configuration',
    
  },
};

export const Topbar = ({
  activePage,
  onToggleMobile,
  surveyCode = 'SURV_001 · Ocean Survey',
}) => {
  const current =
    PAGE_HEADINGS[activePage] || {
      title: 'Mission Overview',
    };

  return (
    <header
      className="
        sticky top-0 z-20
        bg-teal-600 
        border-b border-teal-700
        px-4 md:px-[30px]
        py-4
        flex items-center justify-between
        shadow-[0_2px_10px_rgba(13,148,136,0.18)]
      "
    >

      {/* =========================================================
          LEFT SIDE
          ========================================================= */}
      <div className="flex items-center gap-3">

        {/* Mobile menu */}
        <button
          type="button"
          onClick={onToggleMobile}
          className="
            lg:hidden
            p-2
            rounded-lg
            bg-white/15
            text-white
            border border-white/20
            hover:bg-white/25
            transition-colors
          "
          aria-label="Toggle navigation menu"
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="w-5 h-5"
          >
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>


        {/* Page heading */}
        <div>
          {/* Page title */}
          <div
            className="
              topbar-title
              text-[15px]
              md:text-[18px]
              font-semibold
              mt-0.5
              tracking-[-0.01em]
              text-white
              max-w-[50vw]
              sm:max-w-[60vw]
              md:max-w-none
            "
          >
            {current.title}
          </div>
        </div>
      </div>


      {/* =========================================================
          RIGHT SIDE
          ========================================================= */}
      <div className="flex items-center gap-3 md:gap-4">

        {/* Survey indicator */}
        <div
          className="
            font-mono
            text-[11px]
            font-medium
            text-white
            bg-white/15
            border border-white/25
            px-3
            py-1.5
            rounded-full
            whitespace-nowrap
            backdrop-blur-sm
            shadow-sm
          "
        >
          {surveyCode}
        </div>

      </div>

    </header>
  );
};

export default Topbar;