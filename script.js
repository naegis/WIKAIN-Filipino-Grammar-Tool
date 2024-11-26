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
  
  // Show the paste button again
  document.querySelector('.paste-btn').style.display = 'inline-block';
}

function copyText() {
  // Get the output textarea
  const outputTextarea = document.querySelector('.output textarea');
  
  // Select the text
  outputTextarea.select();
  outputTextarea.setSelectionRange(0, 99999); // For mobile devices
  
  // Copy the text
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

    // Clear previous revisions
    revisionsContainer.innerHTML = '';

    // Create a new div to display changes
    const changesDiv = document.createElement('div');
    changesDiv.style.width = '100%';
    changesDiv.style.height = 'auto'; // Adjust height as needed

    if (data.changes && data.changes.length > 0) {
      // Assuming data.changes is an array of strings with the format you want
      changesDiv.innerHTML = data.changes.map(change => {
        // Example: "kumain - ng + nang mabuti" to "kumain <b>NG ➡️ NANG</b> mabuti"
        return change.replace(/- (\w+) \+ (\w+)/g, '- $1 ➡️ $2');
      }).join('<br>');
    } else {
      changesDiv.innerHTML = 'No changes detected.';
    }

    // Append the div to the revisions container
    revisionsContainer.appendChild(changesDiv);

  } catch (error) {
    console.error('Error:', error);
    outputTextArea.value = 'Nagkaroon ng error habang sinusuri ang grammar.';
  }
}

const text = document.querySelector('.secondtext');

  const textLoad = () => {
    setTimeout(() => {
      text.textContent = "WITH LESS WORRIES"
    }, 0);

    setTimeout(() => {
      text.textContent = "WITH LESS ERRORS"
    }, 4000);

    setTimeout(() => {
      text.textContent = "MORE CONFIDENTLY"
    }, 8000);
  }

  textLoad();
  setInterval(textLoad, 12000);
