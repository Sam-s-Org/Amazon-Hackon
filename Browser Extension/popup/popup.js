// console.log("Starting...");

chrome.tabs.captureVisibleTab(null, {}, function (img) {
  // console.log(img);
  fetch("https://d46a3d290a14e35a0e35c4fb84a7082b.serveo.net/search", {
    method: "POST",
    body: JSON.stringify({
      image: img.split(',')[1]
    }),
    headers: {
      "Content-type": "application/json; charset=UTF-8"
    }
  }).then((response) => {
    // console.log(response.body);
    return response.json()
  })
    .then((data) => {
      // console.log(data)
      var results = data.matches
      document.body.innerHTML = `
        <div class="container">
            <div>
                <div class="heading">Results for you</div>
                <div class="subheading">
                    Check each product page for other buying options. Price and other details may vary based on product size and color.
                </div>
            </div>
            <div class="product-list">`
      results.forEach(element => {
        const [productName, productImageLink, buyButtonLink, rating, price] = element
        const ratingText = (rating === -1) ? "No rating available" : `Rating: ${rating}/5`
        document.body.innerHTML += `
          <div class="product-container">
              <div class="product-image">
                  <img src="${productImageLink}" alt="Product Image">
              </div>
              <div class="product-details">
                  <h2>${productName}</h2>
                  <p>${ratingText}</p>
                  <p class="product-price">${price}</p>
                  <a href="${buyButtonLink}" target="_blank" class="buy-button">Go to Amazon</a>
              </div>
          </div>`
      })
      document.body.innerHTML += `
        </div> <!-- close product-list -->
        </div> <!-- close container -->`
    })
});