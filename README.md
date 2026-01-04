# Experiments with MM-LLMs for document text extraction 

Using gpt-5-mini to convert plain text + images to markdown rich text. A fast alternative to nougat and other proprietary ML models (& free with a GitHub education account if using gpt-5-mini through Copilot!).

## Idea 

- Write a copilot agent + prompt that performs this action (can also get plaintext + image inputs autonomously, but it is probably faster to use a simple script for this).  
- Launch agents through copilot CLI (Would need to optimise number of pages for each agent to process. How many agents can we run in parallel? Do we ever get rate limited when launching too many agents?)
- Create a final 'editor agent' that combines all the markdown files into one, checks consistency & does some light formatting, final touch-ups, and outputs a markdown file for the whole document.
- Would also be nice if we had some way to automatically crop figures from pdf pages.