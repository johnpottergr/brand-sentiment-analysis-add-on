# brand-sentiment-analysis-add-on

- n8n: Add a node to audit_workflow.json:

{
  "name": "Run Sentiment Analysis",
  "type": "n8n-nodes-base.executeCommand",
  "parameters": {
    "command": "python sentiment.py --input input.csv --embeddings"
  }
}


----
- Google Sheets: Import sentiment_output.csv into a “Sentiment” tab; link sentiment_visualization.html.
- Content Cluster Visualization: Use embedding column in sentiment_output.csv as input to visualization.py.

