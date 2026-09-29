// src/context/SurveyContext.jsx
// src/context/SurveyContext.jsx

import { createContext, useContext, useState } from "react";

export const SurveyContext = createContext(null);

export const useSurvey = () => useContext(SurveyContext);

export const SurveyProvider = ({ children }) => {
  const [surveyId, setSurveyId] = useState(null);
  const [analysisId, setAnalysisId] = useState(null);
  const [detections, setDetections] = useState([]);

  return (
    <SurveyContext.Provider
      value={{
        surveyId,
        setSurveyId,
        analysisId,
        setAnalysisId,
        detections,
        setDetections,
      }}
    >
      {children}
    </SurveyContext.Provider>
  );
};