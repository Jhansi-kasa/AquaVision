import React from 'react';

export const Switch = ({ checked = false, onChange }) => {
  return (
    <div
      onClick={() => onChange && onChange(!checked)}
      className={`switch ${!checked ? 'off' : ''}`}
      role="switch"
      aria-checked={checked}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === ' ' || e.key === 'Enter') {
          e.preventDefault();
          onChange && onChange(!checked);
        }
      }}
    />
  );
};
