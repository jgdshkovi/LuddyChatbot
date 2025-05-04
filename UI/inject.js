const script = document.createElement("script");
script.src = chrome.runtime.getURL("dist/index.js"); // no more assets/ here
script.type = "module";
console.log("🔁 Injecting chatbot...");
document.head.appendChild(script);
