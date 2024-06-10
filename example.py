gpt_output = {
  "title": "Google Gemini",
  "author": "Harikrishna Dev",
  "institute": "University of Texas at Dallas",
  "latex_code": "\\documentclass{beamer}\n\\usepackage{lipsum}\n\\title{Google Gemini}\n\\author{Harikrishna Dev}\n\\institute{University of Texas at Dallas}\n\\begin{document}\n\n\\begin{frame}\n  \\titlepage\n\\end{frame}\n\n\\begin{frame}{Overview}\n  \\begin{itemize}\n    \\item Introduction to Google Gemini\n    \\item Key Features\n    \\item Applications\n    \\item Future Prospects\n  \\end{itemize}\n  \\note{This slide provides an overview of the main points we will cover in the presentation.}\n\\end{frame}\n\n\\begin{frame}{Introduction to Google Gemini}\n  \\begin{itemize}\n    \\item Google Gemini is a cutting-edge AI project by Google.\n    \\item Focuses on advanced machine learning and AI techniques.\n  \\end{itemize}\n  \\note{Introduce the audience to Google Gemini and its significance in the AI industry.}\n\\end{frame}\n\n\\begin{frame}{Key Features}\n  \\begin{itemize}\n    \\item Robust Natural Language Processing (NLP) capabilities.\n    \\item Advanced image and video recognition.\n    \\item Enhanced predictive analytics.\n  \\end{itemize}\n  \\note{Discuss the main features that make Google Gemini a powerful tool.}\n\\end{frame}\n\n\\begin{frame}{Applications}\n  \\begin{itemize}\n    \\item Healthcare: Improved diagnostics and personalized medicine.\n    \\item Finance: Better risk assessment and fraud detection.\n    \\item Retail: Enhanced customer experience and inventory management.\n  \\end{itemize}\n  \\note{Explain how Google Gemini is applied in various industries to solve complex problems.}\n\\end{frame}\n\n\\begin{frame}{Future Prospects}\n  \\begin{itemize}\n    \\item Continued advancements in AI research.\n    \\item Integration with other Google services and products.\n    \\item Potential for widespread adoption across multiple sectors.\n  \\end{itemize}\n  \\note{Highlight the future potential and direction of Google Gemini.}\n\\end{frame}\n\n\\begin{frame}{Conclusion}\n  \\begin{itemize}\n    \\item Google Gemini is a transformative AI project.\n    \\item Offers significant benefits across various industries.\n    \\item Promises continued innovation and advancements.\n  \\end{itemize}\n  \\note{Summarize the key points and conclude the presentation.}\n\\end{frame}\n\n\\end{document}"
}


# Original text with \n characters
# text_with_newlines = """
# \\documentclass{beamer}\\n\\title{Google Gemini: A Powerful AI Model}\\n\\author{Your Name}\\n\\institute{Your Institute}\\n\\date{\\today}\\n\\begin{document}\\n\\frame{\\n  \\titlepage\\n}\\n\\frame{\\n  \\frametitle{What is Google Gemini?}\\n  \\begin{itemize}\\n    <item>A state-of-the-art large language model (LLM)</item>\\n    <item>Developed by Google AI</item>\\n    <item>Capable of understanding and responding to complex queries</item>\\n    <item>Can generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way.</item>\\n  \\end{itemize}\\n}\\n\\frame{\\n  \\frametitle{Capabilities of Gemini}\\n  \\begin{itemize}\\n    <item>Text understanding and generation</item>\\n    <item>Machine translation</item>\\n    <item>Code generation</item>\\n    <item>Question answering</item>\\n    <item>and more!\\\\</item>\\n  \\end{itemize}\\n}\\n\\frame{\\n  \\frametitle{Benefits of Using Gemini}\\n  \\begin{itemize}\\n    <item>Improved efficiency and productivity</item>\\n    <item>Enhanced creativity and innovation</item>\\n    <item>Deeper insights and understanding</item>\\n    <item>Potential for various applications across different fields</item>\\n  \\end{itemize}\\n}\\n\\frame{\\n  \\frametitle{Conclusion} \\n  \\begin{itemize}\\n    <item>Google Gemini represents a significant advancement in the field of AI</item>\\n    <item>Holds great promise for the future</item>\\n  \\end{itemize}\\n}\\n\\end{document}\\n
# """

text_with_newlines = gpt_output['latex_code']

# Replace \\n with actual newlines
clean_text = text_with_newlines.replace('\\n', '\n')

# Print or save the cleaned text to a file
print(clean_text)

# Optionally save to a .tex file
with open('presentation.tex', 'w') as file:
    file.write(clean_text)
