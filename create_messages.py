def assistant(content: str):
    return { "role": "assistant", "content": content }

def user(content: str):
    return { "role": "user", "content": content }

def system(content: str):
    return { "role": "system", "content": content }

instructions = """
Detecting Sarcasm in Text: Key Checkpoints and Examples

1. Contradiction Between Literal Meaning and Context:
   Sarcasm often involves saying the opposite of what is meant. Look for statements where the literal meaning contradicts the implied meaning.
   - Example: "Oh great, another meeting. Just what I needed."
     Here, "great" is used sarcastically because meetings are typically not something people look forward to.

2. Exaggeration:
   Sarcasm can involve hyperbolic statements that exaggerate to an absurd degree.
   - Example: "I had the best day ever, I lost my job!"
     The exaggeration of "best day ever" contrasts with the negative event of losing a job, indicating sarcasm.

3. Unexpected Context:
   Statements that seem out of place or overly dramatic for the situation can be sarcastic.
   - Example: "Yeah, because I'm just *so* good at cooking," after burning dinner.
     The unexpected praise for poor performance hints at sarcasm.

4. Punctuation and Emphasis:
   Sarcasm can be indicated by specific punctuation marks, such as quotation marks, italics, or ellipses.
   - Example: "Yeah, because that was a *brilliant* idea..."
     The use of italics and ellipses suggests a sarcastic tone.

5. Contextual Incongruity:
   A statement that is incongruous or mismatched with the situation can be sarcastic.
   - Example: "What a beautiful day," said on a rainy day.
     The statement doesn't match the context of the weather, indicating sarcasm.

6. Understatement:
   Sarcasm can also be conveyed through understatement, where the speaker downplays something significant.
   - Example: "Not bad," after winning a gold medal.
     The understatement of "not bad" contrasts with the significant achievement, implying sarcasm.

7. Positive Adjectives in Negative Situations:
   Using overly positive adjectives to describe a negative situation often indicates sarcasm.
   - Example: "Fantastic, now my computer crashed."
     The word "fantastic" is positive, but the situation is negative, suggesting sarcasm.

8. Cliché Phrases:
   Certain cliché phrases are commonly used sarcastically.
   - Example: "Oh, joy," when faced with an unpleasant task.
     This phrase is often used sarcastically to express the opposite of joy.

9. Tone of Certainty:
   Sarcasm can be conveyed through a tone of certainty or assurance about something that is clearly not true.
   - Example: "Sure, I believe you."
     The certainty in the statement, when paired with an unbelievable claim, indicates sarcasm.

10. Redundancy of Obvious Facts:
    Stating the obvious in an exaggerated manner can be a marker of sarcasm.
    - Example: "Of course the printer jammed. It always jams right when I'm in a hurry."
      The redundancy emphasizes the annoyance and indicates sarcasm.

11. Playful Insults:
    Sarcastic comments can often take the form of playful insults or teasing remarks.
    - Example: "Nice job on that presentation, really top-notch," when the presentation was poorly done.
      The compliment is given in a context where it is clearly not deserved, indicating sarcasm.

12. Literal Statements with Implicit Contradictions:
    Sometimes, a statement is literally true but delivered in a way that implies the opposite meaning.
    - Example: "Well, that was productive," after a meeting where nothing was accomplished.
      The literal truth (the meeting was held) contrasts with the implied meaning (the meeting was unproductive).

13. Hashtags and Emojis:
    Sarcasm on social media often includes specific hashtags or emojis that indicate a sarcastic tone.
    - Example: "Just got a flat tire. #blessed 🙃"
      The hashtag "#blessed" and the upside-down face emoji suggest sarcasm about an unfortunate event.

14. Quotation Marks for Irony:
    Quotation marks can be used to indicate irony or sarcasm.
    - Example: "Had a 'great' time waiting in line for two hours."
      The quotation marks around "great" indicate that the word is being used ironically.

15. Mock Enthusiasm:
    Overly enthusiastic language about mundane or negative events can be a sign of sarcasm.
    - Example: "Yay! Another meeting at 7 AM!"
      The "Yay!" and exclamation mark indicate sarcasm about an early meeting.

16. Short, Snappy Replies:
    Brief responses that use concise language can be sarcastic, especially in the context of a conversation.
    - Example: "Sure, because that makes total sense."
      The brevity and the phrase "total sense" imply sarcasm.

17. Capitalization for Emphasis:
    Unusual capitalization patterns, like all caps, can indicate sarcasm.
    - Example: "I REALLY needed that."
      The use of all caps for "REALLY" emphasizes sarcasm.

18. Tagging and Mentioning:
    Tagging or mentioning others in a way that suggests exaggeration or irony can be a marker of sarcasm.
    - Example: "Thanks @airline for the 'amazing' service."
      Tagging the airline and using quotation marks around "amazing" indicate sarcasm about poor service.

19. Ellipses for Suspense:
    Ellipses can be used to create a sarcastic tone by adding a pause that suggests something contrary to what is stated.
    - Example: "Just what I wanted... another email."
      The ellipsis adds a pause that suggests the opposite of what is stated.

20. Overuse of Positive Adjectives:
    Overusing positive adjectives in a context where they don't fit can indicate sarcasm.
    - Example: "This 'delicious' cafeteria food is just what I needed."
      The adjective "delicious" in quotation marks suggests the opposite.

21. Incongruous Hashtags:
    Using hashtags that don't match the context of the post can signal sarcasm.
    - Example: "Got soaked in the rain. #LivingTheDream"
      The hashtag "#LivingTheDream" contrasts with the unpleasant experience.

22. Self-Deprecating Remarks:
    Sarcastic comments that make fun of the poster themselves can be common on social media.
    - Example: "Guess who forgot their keys again? #genius"
      The hashtag "#genius" indicates sarcasm about forgetting keys.

23. Mimicking Popular Phrases:
    Mimicking popular phrases or memes in a way that subverts their original meaning can indicate sarcasm.
    - Example: "Because who doesn't love working on weekends?"
      The phrase subverts the positive connotation of loving something.

"""


system_guid = "You are an advanced language model designed to analyze and interpret sentences for sarcasm detection, responds in JSON"

system_response_sturcture = 'response only in json {"prediction" : "1 for sarcastic or 0 for not"}'

required_task = "Determine whether the sentence below is sarcastic or not"

rag_template_old = (
            "You are an advanced language model designed to analyze and interpret sentences for sarcasm detection.\n"
            "Determine whether the sentence below is sarcastic or not.\n"
            "Base your answer on the provided context, labeled sentences, ordered by most similar at top and so on\n"
            "Ensure that your answer does not contradict or refute any of the provided context sentences.\n"
            "Your answer should be consistent with the information and labels in the context.\n"
            "If there is a sentence in the context that is similar to the given sentence, you should prioritize its label, even if your initial assessment differs.\n"
            "The sentence:\n{sentence}\n\n"
            "Context:\n\n{context}\n\n"
            "{format_instructions}\n\n"
            "- Do not include newline characters; the entire response should be on one line.\n"
            "- Escape any double quotes inside the string with a backslash (\\\").\n"
            "- Ensure the response contains valid encoded characters.\n"
        )

zero_shot_template_old = (
            "You are an advanced language model designed to analyze and interpret sentences for sarcasm detection.\n"
            "Determine whether the sentence below is sarcastic or not.\n\n"
            "The sentence:\n{sentence}\n\n"
            "{format_instructions}\n\n"
            "Your response must follow these formatting guidelines:\n"
            "- Do not include newline characters; the entire response should be on one line.\n"
            "- Escape any double quotes inside the string with a backslash (\\\").\n"
            "- Ensure the response contains valid encoded characters.\n"
        )

zero_shot_template = (
            "You are an advanced language model designed to analyze and interpret sentences for sarcasm detection.\n"
            "You must determine whether the sentence below is sarcastic.\n\n"
            "The sentence:\n{sentence}\n\n"
            "{format_instructions}\n\n"
        )
rag_template_all = (
            "You are an advanced language model designed to analyze and interpret sentences for sarcasm detection.\n"
            "You must determine whether the sentence below is sarcastic.\n\n"
            "Base your answer on the provided context, labeled sentences, ordered by most similar at top and so on\n"
            "Ensure that your answer does not contradict or refute any of the provided context sentences.\n"
            "Your answer should be consistent with the information and labels in the context.\n"
            "If there is a sentence in the context that is similar to the given sentence, you should prioritize its label, even if your initial assessment differs.\n\n"
            "The sentence:\n{sentence}\n\n"
            "Context:\n{context}\n\n"
            "{format_instructions}\n\n"
        )
#for rillof
rag_template_rillof = (
            "You are an advanced language model designed to analyze and interpret sentences for sarcasm detection.\n"
            "You must determine whether the sentence below is sarcastic.\n\n"
            "Please note that although many sentences include the hashtag #sarcasm, most of them are not actually sarcastic. If you're uncertain, it's better to predict '0'.\n"#add
            "Base your answer on the provided context, labeled sentences, ordered by most similar at top and so on\n"
            "Ensure that your answer does not contradict or refute any of the provided context sentences.\n"
            "Your answer should be consistent with the information and labels in the context.\n"
            "If there is a sentence in the context that is similar to the given sentence, you should prioritize its label, even if your initial assessment differs.\n\n"
            "The sentence:\n{sentence}\n\n"
            "Context:\n{context}\n\n"
            "{format_instructions}\n\n"
        )

#for sarc
rag_template = (
            "You are an advanced language model designed to analyze and interpret sentences for sarcasm detection.\n"
            "You must determine whether the sentence below is sarcastic.\n\n"
            "Please if you're uncertain, it's better to predict '0'.\n"#add
            "Base your answer on the provided context, labeled sentences, ordered by most similar at top and so on\n"
            "Ensure that your answer does not contradict or refute any of the provided context sentences.\n"
            "Your answer should be consistent with the information and labels in the context.\n"
            "If there is a sentence in the context that is similar to the given sentence, you should prioritize its label, even if your initial assessment differs.\n\n"
            "The sentence:\n{sentence}\n\n"
            "Context:\n{context}\n\n"
            "{format_instructions}\n\n"
        )


result_only_format = 'The output should be formatted as the JSON schema:\n{"prediction": "1" if sarcastic or "0" if not}, do not add any more tokens'

def basic_messages(sentence):
  messages=[
    system(system_guid),
    system(system_response_sturcture),
    user(required_task),
    user(f"Sentence: '{sentence}'")
  ]
  return messages


def instructions_messages(sentence):
  messages=[
    system(system_guid),
    system(system_response_sturcture),
    user(f'Determine whether the sentence below is sarcastic or not. Base your answer on this instructions:\n{instructions}'),
    user(f"Sentence: '{sentence}'")
  ]
  return messages

def explain_messages(sentence):
  messages=[
    system(system_guid),
    system("Provide a detailed answer"),
    user("explain for the sentence below why it could be sarcastic and why not, for each option try to convince me"),
    user(f"Sentence: '{sentence}'")
  ]
  return messages

def after_explain_messages(sentence, explanation):
  messages=[
    system(system_guid),
    system(system_response_sturcture),
    user(required_task),
    user(f"Sentence: '{sentence}'"),
    user(f"answer after you read this explanation: {explanation}")
  ]
  return messages

def context_messages(sentence, context):
  messages=[
    system(system_guid),
    system(system_response_sturcture),
    user(f"{required_task}, use the context for this sentence."),
    user(f"Sentence: '{sentence}'"),
    user(f"Context: '{context}'")
  ]
  return messages

def few_shot_messages(sentence, examples):
  messages=[
    system(system_guid),
    system(system_response_sturcture),
    user(f'{required_task} here are few examples: \n {examples}'),
    user(f"Sentence: '{sentence}'")
  ]
  return messages

def dna_messages(sentence):
  system_guid = "You are an advanced language model designed to analyze and interpret sentences for humor, toxicity, irony, politeness detection, responds in JSON"
  system_response_sturcture = 'response only with json {"humor: "1/0", toxicity : "1/0", irony : "1/0", politeness":  "1/0" "} 1 for true, 0 for false'
  required_task = "Determine whether the sentence below is humor, toxicity, irony, politeness or not"
  messages=[
    system(system_guid),
    system(system_response_sturcture),
    user(required_task),
    user(f"Sentence: '{sentence}'")
  ]
  return messages