import React from 'react';
import { Nav } from 'react-bootstrap';

function FormSteps({ step1, step2, step3, step4, step5, step6, step7 }) {
  return (
    <Nav className="justify-content-center mb-4" variant="tabs" style={{ backgroundColor: '#f8f9fa', borderRadius: '8px', padding: '10px' }}>
      <Nav.Item>
        <Nav.Link disabled={!step1} style={{ backgroundColor: step1 ? '#ffc107' : '#f8f9fa', color: step1 ? '#fff' : '#000' }}>Personal Data</Nav.Link>
      </Nav.Item>

      <Nav.Item>
        <Nav.Link disabled={!step2} style={{ backgroundColor: step2 ? '#28a745' : '#f8f9fa', color: step2 ? '#fff' : '#000' }}>Medical Data 1</Nav.Link>
      </Nav.Item>

      <Nav.Item>
        <Nav.Link disabled={!step3} style={{ backgroundColor: step3 ? '#007bff' : '#f8f9fa', color: step3 ? '#fff' : '#000' }}>Medical Data 2</Nav.Link>
      </Nav.Item>

      <Nav.Item>
        <Nav.Link disabled={!step4} style={{ backgroundColor: step4 ? '#dc3545' : '#f8f9fa', color: step4 ? '#fff' : '#000' }}>Medical Data 3</Nav.Link>
      </Nav.Item>

      <Nav.Item>
        <Nav.Link disabled={!step5} style={{ backgroundColor: step5 ? '#6f42c1' : '#f8f9fa', color: step5 ? '#fff' : '#000' }}>Medical Data 4</Nav.Link>
      </Nav.Item>

      <Nav.Item>
        <Nav.Link disabled={!step6} style={{ backgroundColor: step6 ? '#fd7e14' : '#f8f9fa', color: step6 ? '#fff' : '#000' }}>Medical Data 5</Nav.Link>
      </Nav.Item>

      <Nav.Item>
        <Nav.Link disabled={!step7} style={{ backgroundColor: step7 ? '#17a2b8' : '#f8f9fa', color: step7 ? '#fff' : '#000' }}>Additional Questions</Nav.Link>
      </Nav.Item>
    </Nav>
  );
}

export default FormSteps;
