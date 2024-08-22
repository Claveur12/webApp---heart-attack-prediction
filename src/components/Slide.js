import React from 'react';
import '../styles/Slide.css'; 

const Slide = ({ formContent, imageSrc, imageAlt, dummyText }) => {
  return (
    <div className="slide-container">
      <div className="form-section">
        {formContent}
      </div>
      <div className="image-section">
        <div className="image-overlay">
          <p className="dummy-text">{dummyText}</p>
        </div>
        <img src={imageSrc} alt={imageAlt} className="slide-image" />
      </div>
    </div>
  );
};

export default Slide;
