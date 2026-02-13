#these prompts are taken from https://github.com/augustinLib/SPIKE/tree/main/data/embedding_instruction/gte-Qwen1.5-7B-instruct



prompts = {
"aops": {
  "instructions": {
    "query": "Instruct: Given a Math problem, retrieve relevant examples that help answer the problem\nQuery: "
  }
}, 

"biology": {
  "instructions": {
    "query": "Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery: "
  },
  "instructions_long": {
    "query": "Instruct: Given a {task} post, retrieve relevant documents that help answer the post\nQuery: "
  }
},

"earth-science": 
{
  "instructions": {
    "query": "Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery: "
  },
  "instructions_long": {
    "query": "Instruct: Given a {task} post, retrieve relevant documents that help answer the post\nQuery: "
  }
},

"economics": 
{
  "instructions": {
    "query": "Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery: "
  },
  "instructions_long": {
    "query": "Instruct: Given a {task} post, retrieve relevant documents that help answer the post\nQuery: "
  }
},

"leetcode": {
  "instructions": {
    "query": "Instruct: Given a coding problem, retrieve relevant examples that help answer the problem\nQuery: "
  }
},

"pony":  {
  "instructions": {
    "query": "Instruct: Given a {task} question, retrieve relevant passages that help answer the question\nQuery: "
  },
  "instructions_long": {
    "query": "Instruct: Given a {task} question, retrieve relevant documents that help answer the question\nQuery: "
  }
},


"psychology": {
  "instructions": {
    "query": "Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery: "
  },
  "instructions_long": {
    "query": "Instruct: Given a {task} post, retrieve relevant documents that help answer the post\nQuery: "
  }
},

"robotics": {
  "instructions": {
    "query": "Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery: "
  },
  "instructions_long": {
    "query": "Instruct: Given a {task} post, retrieve relevant documents that help answer the post\nQuery: "
  }
},

"stackoverflow": {
  "instructions": {
    "query": "Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery: "
  },
  "instructions_long": {
    "query": "Instruct: Given a {task} post, retrieve relevant documents that help answer the post\nQuery: "
  }
},

"sustainable-living": {
  "instructions": {
    "query": "Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery: "
  },
  "instructions_long": {
    "query": "Instruct: Given a {task} post, retrieve relevant documents that help answer the post\nQuery: "
  }
},



"theoremqa-questions": {
  "instructions": {
    "query": "Instruct: Given a Math problem, retrieve relevant examples that help answer the problem\nQuery: "
  }
},


"theoremqa-theorems": {
  "instructions": {
    "query": "Instruct: Given a Math problem, retrieve relevant theorems that help answer the problem\nQuery: "
  }
}


}






rerank_prompts = {
"aops": {
  "instructions": {
    "query": "Given a Math problem, retrieve relevant examples that help answer the problem"
  }
}, 

"biology": {
  "instructions": {
    "query": "Given a biology post, retrieve relevant passages that help answer the post"
  },
  "instructions_long": {
    "query": "Given a biology post, retrieve relevant documents that help answer the post"
  }
},

"earth-science": 
{
  "instructions": {
    "query": "Given a earth-science post, retrieve relevant passages that help answer the post"
  },
  "instructions_long": {
    "query": "Given a earth-science post, retrieve relevant documents that help answer the post"
  }
},

"economics": 
{
  "instructions": {
    "query": "Given a economics post, retrieve relevant passages that help answer the post"
  },
  "instructions_long": {
    "query": "Given a economics post, retrieve relevant documents that help answer the post"
  }
},

"leetcode": {
  "instructions": {
    "query": "Given a coding problem, retrieve relevant examples that help answer the problem"
  }
},

"pony":  {
  "instructions": {
    "query": "Given a pony question, retrieve relevant passages that help answer the question"
  },
  "instructions_long": {
    "query": "Given a pony question, retrieve relevant documents that help answer the question"
  }
},


"psychology": {
  "instructions": {
    "query": "Given a psychology post, retrieve relevant passages that help answer the post"
  },
  "instructions_long": {
    "query": "Given a psychology post, retrieve relevant documents that help answer the post"
  }
},

"robotics": {
  "instructions": {
    "query": "Given a robotics post, retrieve relevant passages that help answer the post"
  },
  "instructions_long": {
    "query": "Given a robotics post, retrieve relevant documents that help answer the post"
  }
},

"stackoverflow": {
  "instructions": {
    "query": "Given a stackoverflow post, retrieve relevant passages that help answer the post"
  },
  "instructions_long": {
    "query": "Given a stackoverflow post, retrieve relevant documents that help answer the post"
  }
},

"sustainable-living": {
  "instructions": {
    "query": "Given a sustainable-living post, retrieve relevant passages that help answer the post"
  },
  "instructions_long": {
    "query": "Given a sustainable-living post, retrieve relevant documents that help answer the post"
  }
},



"theoremqa-questions": {
  "instructions": {
    "query": "Given a Math problem, retrieve relevant examples that help answer the problem"
  }
},


"theoremqa-theorems": {
  "instructions": {
    "query": "Given a Math problem, retrieve relevant theorems that help answer the problem"
  }
}


}