# Recommendation API curl Examples

All commands assume the API is running locally at `http://localhost:8080`. Adjust `BASE_URL` if you deploy elsewhere.

```bash
BASE_URL=${BASE_URL:-http://localhost:8080}
```

## 1. Basic Recommendation With Explicit Interactions

```bash
curl -X POST "$BASE_URL/v1/recommend" \
  -H 'Content-Type: application/json' \
  -d '{
        "user_id": "demo-user",
        "k": 12,
        "candidate_k": 300,
        "interactions": [
          {"item_id": "68ba0fdda97aed5b97de45af", "event": "product_click", "timestamp": "2025-10-01T23:36:59.894Z"},
          {"item_id": "68c34759a97aed5b97c4ce1d", "event": "buy_click", "timestamp": "2025-10-01T23:37:59.490Z"}
        ]
      }' | jq .
```

```bash
{
  "user": {
    "build": "inline"
  },
  "items": [
    {
      "item_id": "68ba0fdda97aed5b97de45af",
      "rank": 1,
      "score": 0.893145921088386,
      "sim": 0.9884978532791138,
      "title": "Christabel embellished minidress",
      "brand": "RIXO",
      "price": 745.0,
      "color": "white",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/8d/P01105838.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "687fb196681d8584dd881562",
      "rank": 2,
      "score": 0.3469015325160782,
      "sim": 0.984998881816864,
      "title": "Embellished halterneck minidress",
      "brand": "Rotate",
      "price": 415.0,
      "color": "white",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/db/P01086822.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "6893fc2eb559e26af7d1c881",
      "rank": 3,
      "score": 0.29560722852661186,
      "sim": 0.984502375125885,
      "title": "Le Sable sequined minidress",
      "brand": "Staud",
      "price": 1245.0,
      "color": "white",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/72/P01074610.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "67d386783ce6d6673d1ac9b4",
      "rank": 4,
      "score": 0.2510186789850274,
      "sim": 0.980505108833313,
      "title": "Beaded minidress",
      "brand": "Rotate",
      "price": 280.0,
      "color": "silver",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/dc/P01040380.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "67636f03b5466f0fa06cb9fe",
      "rank": 5,
      "score": 0.23721975001811554,
      "sim": 0.9800574779510498,
      "title": "Star Fish sequined minidress",
      "brand": "RIXO",
      "price": 445.0,
      "color": "multicoloured",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/87/P00988855.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "68476eaefa2061ee6045412f",
      "rank": 6,
      "score": 0.2247194420793612,
      "sim": 0.982225239276886,
      "title": "Droplet embellished beaded minidress",
      "brand": "Clio Peppiatt",
      "price": 2685.0,
      "color": "white",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/77/P01053864.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "67081a5fd4a26ba452b9bde9",
      "rank": 7,
      "score": 0.21816938259003493,
      "sim": 0.9858893156051636,
      "title": "Bridal embellished halterneck minidress",
      "brand": "Rotate",
      "price": 332.0,
      "color": "white",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/4b/P00990948.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "66b082d71cf2893fba9804f7",
      "rank": 8,
      "score": 0.20386138231163706,
      "sim": 0.9860173463821411,
      "title": "Bow-detail sequined minidress",
      "brand": "self-portrait",
      "price": 480.0,
      "color": "white",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/83/P00978735.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "675bb137b5466f0fa0eb6b8b",
      "rank": 9,
      "score": 0.19594874238511717,
      "sim": 0.9842166304588318,
      "title": "Sequined floral-appliqué minidress",
      "brand": "Magda Butrym",
      "price": 1491.0,
      "color": "silver",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/33/P00975535.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "6853536335410b68268de2db",
      "rank": 10,
      "score": 0.19239358170098167,
      "sim": 0.986792802810669,
      "title": "Bridal Meha feather-trimmed minidress",
      "brand": "Rebecca Vallance",
      "price": 1640.0,
      "color": "white",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/ff/P01039615.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "67051da7d4a26ba4524b1b7e",
      "rank": 11,
      "score": 0.18581748541615964,
      "sim": 0.9801194667816162,
      "title": "Sequined one-shoulder minidress",
      "brand": "Givenchy",
      "price": 2789.0,
      "color": "silver",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/fb/P00692651.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "670532a1d4a26ba4524d602a",
      "rank": 12,
      "score": 0.18260502713463583,
      "sim": 0.9824072122573853,
      "title": "Bridal Grace sequined minidress",
      "brand": "Rotate",
      "price": 211.0,
      "color": "pink",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/02/P00804412.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    }
  ],
  "meta": {
    "k": 12,
    "candidate_k": 300
  }
}
```

## 2. Cold-Start Request (No Interaction Payload)

```bash
curl -X POST "localhost:8080/v1/recommend" \
  -H 'Content-Type: application/json' \
  -d '{
        "user_id": "new-user",
        "k": 8,
        "candidate_k": 200
      }' | jq .
```

```bash
{
  "user": {
    "build": "cache"
  },
  "items": [
    {
      "item_id": "67a2a8cdb95336b784765a81",
      "rank": 1,
      "score": 0.000013579809039056138,
      "sim": 0.9708367586135864,
      "title": "Trey ruffled ribbed-knit gown",
      "brand": "Altuzarra",
      "price": 840.0,
      "color": "brown",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/a8/P01023333.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "67053a92d4a26ba4524e3f3d",
      "rank": 2,
      "score": 0.000013575728022787342,
      "sim": 0.9723259210586548,
      "title": "Ruffled asymmetrical midi dress",
      "brand": "Alessandra Rich",
      "price": 779.0,
      "color": "black",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/1a/P00818355.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "6704a7d8d4a26ba4523db3c9",
      "rank": 3,
      "score": 0.000013565832399075786,
      "sim": 0.9709049463272095,
      "title": "Ruched silk midi dress",
      "brand": "Alessandra Rich",
      "price": 820.0,
      "color": "black",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/ba/P00694741.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "66a9f0342db8a68331e695ef",
      "rank": 4,
      "score": 0.000013544030094312661,
      "sim": 0.971092700958252,
      "title": "Cutout maxi dress",
      "brand": "Johanna Ortiz",
      "price": 875.0,
      "color": "white",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/bc/P00950204.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "6704b316d4a26ba4523f2659",
      "rank": 5,
      "score": 0.000013533154677709184,
      "sim": 0.9710174798965454,
      "title": "Gingham-print halterneck dress",
      "brand": "Acne Studios",
      "price": 805.0,
      "color": "multicoloured",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/20/P00778376.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "6648529b16f50168ab6ebbdf",
      "rank": 6,
      "score": 0.000013514014774720918,
      "sim": 0.971325159072876,
      "title": "Draped asymmetric urania midi dress",
      "brand": "Rabanne",
      "price": 792.0,
      "color": "neutrals",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/62/P00932301.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "665a63d9fc11a2f59dade72a",
      "rank": 7,
      "score": 0.00001351266204059786,
      "sim": 0.9707424640655518,
      "title": "Saralien gathered jersey maxi dress",
      "brand": "Altuzarra",
      "price": 870.0,
      "color": "orange",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/3f/P00950004.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    },
    {
      "item_id": "6621cb27e8bcdaefb99470ca",
      "rank": 8,
      "score": 0.000013510232102358376,
      "sim": 0.970355749130249,
      "title": "Sequined jacquard maxi dress",
      "brand": "Acne Studios",
      "price": 840.0,
      "color": "grey",
      "image": "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/27/P00921852.jpg",
      "explanations": [
        "Similarity to your history"
      ]
    }
  ],
  "meta": {
    "k": 8,
    "candidate_k": 200
  }
}
```
