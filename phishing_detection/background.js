chrome.runtime.onInstalled.addListener(() => {
    console.log("Phishing Detection Extension Installed");
  });
  
  chrome.action.onClicked.addListener(async (tab) => {
    // Execute content script on the active tab
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['content.js']
    });
  });
  