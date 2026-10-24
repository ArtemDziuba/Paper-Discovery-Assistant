#### 1. Define the Task 
The Paper Discovery Assistant (PDA) is a tool for fast search of relevant papers on arXiv to the user query. 
The application should:
1. Interpret query and identify the key research concepts.
2. Search arXiv for potentially relevant papers.
3. Retrieve their titles, abstracts, authors, dates, categories, and links.
4. Rank the papers according to their relevance to your question.
5. Display a score, for each paper, along with an explanation of the match.
6. Let you open the original paper or abstract.
7. Generate a literature overview based on the retrieved papers, with citations linking back to the sources.

#### 2. Prepare the Data 

#### 3. Describe the Components 
Draw a diagram that shows the data flow. Describe the purpose, input, and output of each  component. Identify model calls and external tool calls. Explain how one input passes through  the complete system. You can use LLM for this task, but you must control the process.  

#### 4. Build the Prototype 
Give the AI coding assistant a clear request and make sure you manage the instructions to your agent not only through cycles of prompting. You must create agents md file or convention md file or any other way to handle and steer agentic behaviour and context. Follow Crisp-dm by instructing your coding agent to do it properly, put instructions into the agents md or similar file.  

#### 5. Evaluate the System 
Evaluation is required for every component of the project. It can be also done by an LLM but you must be able to distinguish between high and low scores, think of how to improve it and explain why scores are not as good as they could be (if the case). The selection of metrics will also be judged, so make sure you choose proper metrics.