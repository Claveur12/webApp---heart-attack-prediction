
import React from 'react';
import { Container, Col, Row, Form, Image } from 'react-bootstrap';
import heartAttackImg from '../images/heart_attack_illustration.webp';
import '../styles/PredictionForm.css';

const MedicalData1 = ({ formData, handleChange }) => (
  <Container className="mt-4 medical-data">
    <h2>Personal Data</h2>
    <Row className='column-field-data'>
      {/* Left Column: Form Inputs */}
      <Col md={6} className='column-field-data'>
      
        <Form className='form-group-data'>
          <Form.Group controlId="formFirstName" className='form-group'>
            <Form.Label>First Name</Form.Label>
            <Form.Control
              type="text"
              name="firstName"
              value={formData.firstName}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formLastName" className='form-group'>
            <Form.Label>Last Name</Form.Label>
            <Form.Control
              type="text"
              name="lastName"
              value={formData.lastName}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formHeight" className='form-group'>
            <Form.Label>Height</Form.Label>
            <Form.Control
              type="number"
              name="height"
              value={formData.height}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group controlId="formWeight" className='form-group'>
            <Form.Label>Weight</Form.Label>
            <Form.Control
              type="number"
              name="weight"
              value={formData.weight}
              onChange={handleChange}
              required
            />
          </Form.Group>
        </Form>
      </Col>

      {/* Right Column: Image */}
      <Col md={6} className='column-field-data'>
        <Row>
          <Col>
            <Image
              src={heartAttackImg}
              alt="Personal Data"
              fluid
            className="d-block w-100" 
            />
          </Col>       
        </Row>
        <Row className='heart-attack-comment'>
          <Col>
          <p>
          <h5>What is a Heart Attack? </h5><b/>         
          <span className='heart-text-comment'>A heart attack occurs when a blocked artery 
            prevents blood and oxygen from reaching the heart.</span>
          </p>          
          </Col>
        </Row>
      </Col>
    </Row>
  </Container>
);

export default MedicalData1;
