function pasteText() {
  // Hide the paste button once clicked
  const pasteButton = document.querySelector('.paste-btn');
  pasteButton.style.display = 'none'; // Hide the paste button
  
  // Paste the clipboard content into the textarea
  navigator.clipboard.readText().then(text => {
    document.querySelector('textarea').value = text;
  });
}

// Function to clear the textarea and show the paste button again
function clearText() {
  // Clear the textarea content
  document.querySelector('textarea').value = '';
  

  document.querySelector('.paste-btn').style.display = 'inline-block';
  

  document.getElementById('output-text').value = '';

  const revisionsContainer = document.querySelector('.revisions');
  revisionsContainer.innerHTML = `

    <span class="revisionsp1">
      Your revisions will appear here.
    </span>
    <span class="revisionsp2">
      <br>Start writing to see your corrections.
    </span>
  `;
}

function copyText() {

  const outputTextarea = document.querySelector('.output textarea');
  

  outputTextarea.select();
  outputTextarea.setSelectionRange(0, 99999);
  

  navigator.clipboard.writeText(outputTextarea.value)
      .then(() => {
          // Optional: Add some visual feedback
          const copyIcon = document.querySelector('.output .delete-icon');
          copyIcon.style.opacity = '0.5';
          setTimeout(() => {
              copyIcon.style.opacity = '1';
          }, 200);
      })
      .catch(err => {
          console.error('Failed to copy text: ', err);
      });
}


async function checkGrammar() {
  const inputText = document.getElementById('input-text').value;
  const outputTextArea = document.getElementById('output-text');
  const revisionsContainer = document.querySelector('.revisions');

  try {
    const response = await fetch('http://127.0.0.1:5000/check-grammar', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: inputText })
    });

    const data = await response.json();
    outputTextArea.value = data.corrected_text;


    revisionsContainer.innerHTML = '';


    const changesDiv = document.createElement('div');
    changesDiv.style.width = '100%';
    changesDiv.style.height = 'auto'; 

    if (data.changes && data.changes.length > 0) {
   
      let tableContainer = '<div style="max-height: 150px; max-width: 100%; overflow-y: auto;">';
      tableContainer += '<table style="width: 100%; border-collapse: collapse; table-layout: fixed;">';
      tableContainer += '<tr><th style="width: 50%; border: 1px solid #ddd;">Original</th><th style="width: 50%; border: 1px solid #ddd;">Corrected</th></tr>';
      
      data.changes.forEach(change => {
        // Add null check and safer parsing
        try {
          const match = change.match(/Original: '([^']+)' Corrected: '([^']+)'/);
          if (match && match.length >= 3) {
            const [, original, corrected] = match;
            tableContainer += `<tr><td style="padding: 5px; border: 1px solid #ddd;">${original}</td><td style="padding: 5px; border: 1px solid #ddd;">${corrected}</td></tr>`;
          } else {
            console.warn('Invalid change format:', change);
          }
        } catch (parseError) {
          console.warn('Failed to parse change:', change, parseError);
        }
      });
       
      tableContainer += '</table></div>'; 
      changesDiv.innerHTML = tableContainer;
    } else {
      changesDiv.innerHTML = 'No changes detected.';
    }

    revisionsContainer.appendChild(changesDiv);

  } catch (error) {
    console.error('Error:', error);
    outputTextArea.value = 'Nagkaroon ng error habang sinusuri ang grammar. (JS)';
  }
}

function handleKeyDown(event) {
  if (event.key === 'Enter') {
    const pasteButton = document.querySelector('.paste-btn');
    if (pasteButton) {
      pasteButton.style.display = 'none';
    }
    
    event.preventDefault(); 
    checkGrammar(); 
  }
}

// Ensure the event listener is added to the input element
const inputArea = document.getElementById('input-text');
if (inputArea) {
  inputArea.addEventListener('keydown', handleKeyDown);
}

function hidePasteButtonOnPaste() {
  const inputArea = document.getElementById('input-text');
  const pasteButton = document.querySelector('.paste-btn');

  if (inputArea && pasteButton) {
    inputArea.addEventListener('paste', () => {
      pasteButton.style.display = 'none';
    });
  }
}


hidePasteButtonOnPaste();