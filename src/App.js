// import React from 'react';
// import Header from './components/Header';
// import Footer from './components/Footer';
// import PredictionForm from './screens/PredictionForm';
// import 'bootstrap/dist/css/bootstrap.min.css';


// const App = () => {
//   return (
//     <div>
//       <Header />
//       <div className="container mt-4">
//         <PredictionForm />
//       </div>
//       <Footer />
//     </div>
//   );
// };

// export default App;


import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import PredictionForm from './screens/PredictionForm';
import 'bootstrap/dist/css/bootstrap.min.css';
import ResultPrediction from './screens/ResultPrediction';

const App = () => {
  return (
    <Router>
      <Header />
      <div className="container mt-4">
        <Routes>
          <Route path="/" element={<PredictionForm />} />
          <Route path="/result" element={<ResultPrediction />} />
        </Routes>
      </div>
      <Footer />
    </Router>
  );
};

export default App;
