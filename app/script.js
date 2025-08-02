const predictButton = document.getElementById("predict-btn");
const resultDiv = document.getElementById("results");
const RLpredictButton = document.getElementById("predict-btn-RL")
const RLresults = document.getElementById("resultsRL")


predictButton.addEventListener("click", async () => {
  const asset = document.getElementById("asset").value;
  const timestamp = document.getElementById("timestamp").value;
    resultDiv.innerHTML = `<div class="spinner"></div>`;

  try {
    const response = await fetch(`http://localhost:8000/predict?asset=${asset}&timestamp=${encodeURIComponent(timestamp)}`);
    // const response = await fetch(`http://localhost:8000/predict?asset=${asset}`);
    const data = await response.json();

    resultDiv.innerHTML = `<h3 style="margin-bottom: 10px">Predicted OHLCV Data:</h3>`;
    data.forEach((item) => {
      resultDiv.innerHTML += `
        <div  style ="line-height:1.5" class="prediction-card">
         <div><strong>Date:</strong> ${item.date}</div><br/>
          <div><strong>Open:</strong> ${item.open.toFixed(2)}</div><br/>
          <div><strong>High:</strong> ${item.high.toFixed(2)}</div><br/>
          <div><strong>Low:</strong> ${item.low.toFixed(2)}</div><br/>
          <div><strong>Close:</strong> ${item.close.toFixed(2)}</div><br/>
          <div><strong>Volume:</strong> ${item.volume.toFixed(2)}</div><br/><br/>
        </div>
      `;
      document.querySelector(".container").style.height = "730px";

    });

  } catch (err) {
    resultDiv.innerHTML = `<div style="display:flex ; align-items:center; justify-content:center" ><p style="color:red">Error fetching prediction: ${err.message}</p></div>`;
  }
});


// RLpredictButton.addEventListener("click", async () => {
//   const asset = document.getElementById("assetRL").value;
//   const timestampRL = document.getElementById("timestampRL").value;
//   const balance =document.getElementById("balance").value;
//   const holdings =document.getElementById("holdings").value;
//     RLresults.innerHTML = `<div class="spinner"></div>`;

//   try {
//     const response = await fetch(`http://localhost:8000/get-action?datetime_str=${encodeURIComponent(timestampRL)}Z&balance=${balance}&holdings=${holdings}`);
//     // const response = await fetch(`http://localhost:8000/predict?asset=${asset}`);
//     const data = await response.json();
  
//     RLresults.innerHTML = `<h3 style="margin-bottom: 10px">Predicted Action:</h3>`;
//     data.forEach((item) => {
//       console.log(item.action);
//       RLresults.innerHTML += `
//         <div  style ="line-height:1.5" class="prediction-card-RL">
//           <strong>Action:</strong> ${item.action}<br/>
//         </div>
//       `;
//       document.querySelector(".RL-prediction").style.height = "400px";

//     });

//   } catch (err) {
//     RLresults.innerHTML = `<div style="display:flex ; align-items:center; justify-content:center" ><p style="color:red">Error fetching prediction: ${err.message}</p></div>`;
//   }
// });
RLpredictButton.addEventListener("click", async () => {
  const asset = document.getElementById("assetRL").value;
  const timestampRL = document.getElementById("timestampRL").value;
  const balance = document.getElementById("balance").value;
  const holdings = document.getElementById("holdings").value;
  const MarketCondition = document.getElementById("assetMarket").value;

  RLresults.innerHTML = `<div class="spinner"></div>`;

  try {
    const response = await fetch(`http://127.0.0.1:8000/get-action?datetime_str=${encodeURIComponent(timestampRL)}Z&balance=${balance}&holdings=${holdings}&market_condition=${MarketCondition}&crypto_type=${asset}`);
    const data = await response.json();
    console.log(data);

    RLresults.innerHTML = `<h3 style="margin-bottom: 10px">Predicted Action:</h3>`;
    RLresults.innerHTML += `
      <div style="line-height:1.5" class="prediction-card-RL">
       <div> <strong>Action:</strong> ${data.action}</div><br/>
       <div> <strong>Units To ${data.action}:</strong> ${data.units_to}</div><br/>
      </div>
    `;

    document.querySelector(".RL-prediction").style.height = "750px";

  } catch (err) {
    RLresults.innerHTML = `<div style="display:flex; align-items:center; justify-content:center">
      <p style="color:red">Error fetching prediction: ${err.message}</p>
    </div>`;
  }
});
