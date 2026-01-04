# Experiments with MM-LLMs for document text extraction 

Using gpt-5-mini to convert plain text + images to markdown rich text. A fast alternative to [nougat](https://github.com/facebookresearch/nougat) and other proprietary ML models (& free with a GitHub education account if using gpt-5-mini through Copilot!).

## Idea 

- Write a copilot agent + prompt that performs this action (can also get plaintext + image inputs autonomously, but it is probably faster to use a simple script for this).  
- Launch agents through copilot CLI (Would need to optimise number of pages for each agent to process. How many agents can we run in parallel? Do we ever get rate limited when launching too many agents?)
- Create a final 'editor agent' that combines all the markdown files into one, checks consistency & does some light formatting, final touch-ups, and outputs a markdown file for the whole document.
- Would also be nice if we had some way to automatically crop figures from pdf pages. Preliminary tests with gpt 5-mini and 5.2 show MMLLMs are pretty bad at figuring out where the edges of a plot are, but combination with computer viz models and right text + image context might help.
  - Identifying plot axes in an image should be textbook CNN work in principle (translation invariant ID of very characteristic shapes) or even template [matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html). 
- Benchmark performance against nougat running on Perlmutter (maybe we can ask Joerg for some GPU hours specifically for this so we can use timed quad-GPU nodes instead of single-GPU login nodes). 
