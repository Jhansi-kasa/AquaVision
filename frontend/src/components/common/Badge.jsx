import React from 'react';

export const Badge = ({ level = 'low', label }) => {
  const normalizedLevel = level.toLowerCase();
  const displayLabel = label || normalizedLevel.toUpperCase();

  return (
    <span className={`badge ${normalizedLevel}`}>
      <i className="bdot"></i>
      {displayLabel}
    </span>
  );
};
