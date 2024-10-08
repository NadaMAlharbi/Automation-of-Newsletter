from flask import Flask, render_template, request
import re
from bs4 import BeautifulSoup
from langchain.chains.combine_documents import create_stuff_documents_chain
from LangChain_Automation import translate_text, model, summarization_prompt, get_documents_from_web, translate_title

# Initialize the Flask web application
app = Flask(__name__)

# Function to split the summary into sentences based on punctuation
def split_summary_into_sentences(summary):
    """
    Split the summary into sentences based on periods, commas, and other delimiters.
    """
    sentences = re.split(r'[.!?،؛]', summary)  # Use punctuation marks as delimiters
    return [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences and strip whitespace

# Function to highlight keywords in the article based on sentences in the summary
def highlight_keywords_in_article(article, summary):
    # Clean up the article and summary to remove any HTML tags
    article_text = BeautifulSoup(article, 'html.parser').get_text()
    summary_text = BeautifulSoup(summary, 'html.parser').get_text()

    # Split the summary into sentences
    summary_sentences = split_summary_into_sentences(summary_text)

    # List of soft, pastel colors to alternate between for highlighting
    colors = ['#FFFACD', '#D3F8E2', '#D4F1F4', '#FFE4E1', '#FFDAB9']  # LemonChiffon, MintCream, LightCyan, MistyRose, PeachPuff

    # Split the article into words with their positions
    article_words = [(word, i) for i, word in enumerate(re.findall(r'\w+', article_text))]

    highlighted_article_words = article_words[:]

    # Iterate over each sentence in the summary and highlight matching words in the article
    for sentence_index, sentence in enumerate(summary_sentences):
        # Split the sentence into words
        sentence_words = [(word, i) for i, word in enumerate(re.findall(r'\w+', sentence))]

        # Use a different color for each sentence
        color = colors[sentence_index % len(colors)]

        # Highlight the keywords for this sentence in the article
        highlighted_article_words = highlight_keywords_by_position_colored(
            article_words=highlighted_article_words,
            sentence_words=sentence_words,
            color=color,
            window_size=5  # Adjust window size for context matching
        )

    # Rejoin the highlighted words into the full article text
    highlighted_article = ' '.join([word for word, _ in highlighted_article_words])
    return highlighted_article

# Function to highlight keywords with different colors for each sentence
def highlight_keywords_by_position_colored(article_words, sentence_words, color, window_size=5):
    """
    Highlights keywords in the article for a specific sentence using a specific color.
    """
    used_keywords = set()  # To track which word sequences have already been highlighted

    # Helper function to get a sequence of words (with window_size length) around a given position
    def get_word_sequence(words, pos, window_size):
        start = max(0, pos - (window_size // 2))  # Start from previous words
        end = min(len(words), pos + (window_size // 2) + 1)  # Include next words
        return [w.lower() for w, _ in words[start:end]]  # Return list of lowercase words for matching

    # Iterate through each word in the article with its position
    for i, (word, pos) in enumerate(article_words):
        # Get the sequence of words in the article (current word + surrounding words)
        article_sequence = get_word_sequence(article_words, i, window_size)

        # Iterate through the sentence words to find a matching sequence
        for j, (sentence_word, sentence_pos) in enumerate(sentence_words):
            # Get the sequence of words in the sentence (current word + surrounding words)
            sentence_sequence = get_word_sequence(sentence_words, j, window_size)

            # Use set-based similarity for partial matching
            matching_words = set(article_sequence).intersection(set(sentence_sequence))
            match_ratio = len(matching_words) / window_size  # النسبة المئوية للتطابق

            # If match ratio is above the threshold and sequence hasn't been highlighted before
            if match_ratio >= 0.5 and tuple(article_sequence) not in used_keywords:
                # Highlight the word with the given color and mark the sequence as used
                article_words[i] = (f'<span style="background-color:{color}; padding: 0.2em;">{word}</span>', pos)
                used_keywords.add(tuple(article_sequence))  # Add sequence to used keywords
                break

    return article_words

# Route for processing the URL and rendering result
@app.route('/process_url', methods=['POST'])
def process_url():
    article_url = request.form['article_url']
    highlight_option = request.form.get('highlight_option', 'yes')  # Default to 'yes'

    # Extract the article content using the provided URL
    docs = get_documents_from_web(article_url)
    full_article = "\n".join([doc.page_content for doc in docs])

    # Create a chain to summarize the document using LangChain
    chain = create_stuff_documents_chain(
        llm=model,  # Pass the language model (LLM) to the chain
        prompt=summarization_prompt  # Pass the summarization prompt
    )
    # Get the summary of the article
    response = chain.invoke({"context": docs})
    title_and_summary = response.split("\n", 1)
    title = title_and_summary[0].strip()
    summary = title_and_summary[1].strip() if len(title_and_summary) > 1 else ""

    # Translate and format the title according to the rules
    translated_title = translate_title(title)

    # Check the highlight option selected by the user
    if highlight_option == 'yes':
        highlighted_article = highlight_keywords_in_article(full_article, summary)
    else:
        highlighted_article = full_article  # Show the article without highlights

    # Translate the summary to Arabic
    translated_summary = translate_text(summary, target_language="ar")

    return render_template('result.html', title=title, translated_title=translated_title, summary=summary,
                           highlighted_article=highlighted_article, translated_summary=translated_summary, article_url=article_url)

# Additional route to toggle highlight option
@app.route('/toggle_highlight', methods=['POST'])
def toggle_highlight():
    article_url = request.form['article_url']
    highlight_option = request.form['highlight_option']

    # Redirect to process_url with the updated highlight option
    return process_url()

# Main input page
@app.route('/')
def index():
    return render_template('form.html')

# Run the application
if __name__ == '__main__':
    app.run(debug=True, port=5002)