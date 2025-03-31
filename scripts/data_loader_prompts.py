PROMPT_TEMPLATE = {
    "SingleQA": [
        {"role": "system", "content": "Your task is to answer the user’s question based on a long context, which consists of many bios. Output the answer only. Don't explain or output other things."},
        {"role": "user", "content": "Context: {given_context}\n\nQuestion: {question}"},
        {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
    ],
    "MultiQA": [
        {"role": "system", "content": "Your task is to answer all the user’s questions based on a long context, which consists of many bios. Output only the answers for each question sequentially. Don't explain or output other things."},
        {"role": "user", "content": "Context: {given_context}\n\nThe Questions are as follows:\n{question}\nAnswer each question in sequence."},
        {"role": "assistant", "content": "Based on the provided context, the answer is "}
    ],
    "SingleQAForBase": "Context: {given_context}\n\nQuestion: {question}\n\nBased on the provided context, {question_prefix}",
    "MultiQAForBase": "Context: {given_context}\n\nQuestion: {question}\n\nBased on the provided context, {question_prefix}",
    "FewShotSingleQA": [
        {"role": "system", "content": "Following the format of the examples, your task is to answer the user’s question based on a long context, which consists of many bios. Output the answer only. Don't explain or output other things."},
        {"role": "user", "content": "Context: {given_context}\n\n{examples}\n\nQuestion: {question}\nAnswer: "},
        {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
    ],
    "FewShotMultiQA": [
        {"role": "system", "content": "Following the format of the examples, your task is to answer all the user’s questions based on a long context, which consists of many bios. Output only the answers for each question sequentially. Don't explain or output other things."},
        {"role": "user", "content": "Context: {given_context}\n\n{examples}\n\n{question}\nAnswers:\n"},
        {"role": "assistant", "content": "Based on the provided context, "}
    ],
    "FewShotSingleQAForBase": "Context: {given_context}\n\n{examples}\n\nQuestion: {question} \nAnswer: Based on the provided context, {question_prefix}",
    "FewShotMultiQAForBase": "Context: {given_context}\n\n{examples}\n\n{question}\nAnswers:\nBased on the provided context,",
    "Rank": [
        {"role": "system", "content": "Your task is to rank the users based on their bios in a long context. Output the answer only. Don't explain or output other things."},
        {"role": "user", "content": "Context: {given_context}\n\nQuestion: {question}"},
        {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
    ],
    "RankForBase": "Context: {given_context}\n\nQuestion: {question}\n\nBased on the provided context, {question_prefix}",
    "FewShotRank": [
        {"role": "system", "content": "Following the format of the examples, your task is to rank the users based on their bios in a long context. Output the answer only. Don't explain or output other things."},
        {"role": "user", "content": "Context: {given_context}\n\n{examples}\n\nQuestion: {question}"},
        {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
    ],
    "FewShotRankForBase": "Context: {given_context}\n\n{examples}\n\nQuestion: {question} \nAnswer: Based on the provided context, {question_prefix}",
    "Calculation": [
        {"role": "system", "content": "Your task is to calculate the difference between the users' age based on a long context, which consists of many bios. Output the answer only. Don't explain or output other things."},
        {"role": "user", "content": "Context: {given_context}\n\nQuestion: {question}"},
        {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
    ],
    "CalculationForBase": "Context: {given_context}\n\nQuestion: {question}\n\nBased on the provided context, {question_prefix}",
    "FewShotCalculation": [
        {"role": "system", "content": "Your task is to calculate the age difference of the given people based on the given instruction from a long context containing multiple bios."},
        {"role": "user", "content": "Context: {given_context}\n\n{examples}\n\nQuestion: {question}"},
        {"role": "assistant", "content": "Answer: Based on the provided context, "}
    ],
    "FewShotCalculationForBase":"Context: {given_context}\n\n{examples}\n\nQuestion: {question} \nAnswer: Based on the provided context",
    "TwoDiff": [
        {"role": "system", "content": "Your task is to follow the instruction and provide the names of the people based on a long context, which consists of many bios."},
        {"role": "user", "content": "Context: {given_context}\n\nQuestion: {question}"},
        {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
    ],
    "TwoDiffForBase": "Context: {given_context}\n\nQuestion: {question}\n\nBased on the provided context, {question_prefix}",
    "FewShotTwoDiff": [
        {"role": "system", "content": "Your task is to find the names of people based on the given instruction from a long context containing multiple bios. Follow the format provided in the examples closely and give the final answer."},
        {"role": "user", "content": "Context: {given_context}\n\n{examples}\n\nQuestion: {question}"},
        {"role": "assistant", "content": "Answer:"}
    ],
    "FewShotTwoDiffForBase": "Context: {given_context}\n\n{examples}\n\nQuestion: {question} \nAnswer:",
    "Multihop":  [
        {"role": "system", "content": "Your task is to answer the user’s question based on a long context, which consists of many bios. Output the answer only. Don't explain or output other things."},
        {"role": "user", "content": "Context: {given_context}\n\nQuestion: {question}"},
        {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
    ],
    "MultihopForBase":"Context: {given_context}\n\nQuestion: {question}\n\nBased on the provided context, {question_prefix}",
    "FewShotMultihop":[
        {"role": "system", "content": "Following the format of the examples, your task is to answer the user’s question based on a long context, which consists of many bios."},
        {"role": "user", "content": "Context: {given_context}\n\n{examples}\n\nQuestion: {question}\nAnswer: "},
        {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
    ],
    "FewShotMultihopForBase":"Context: {given_context}\n\n{examples}\n\nQuestion: {question} \nAnswer:",
    "TextGeneration":[
        {"role": "system", "content": "Given a long context consisting many bios and some related incompleted information, your task is to generate a coherent story incorporating all given information from a given long context. You should retrieve required information from the context and ensure that every piece of the retrieved knowledge is accurately and clearly represented in the stroy. For example, if Bob's bio states he loves basketball, explicitly mention his passion for the basketball. You can ignore the information of the other people."},
        {"role": "user", "content": "Context: {given_context}\n\nGenerate a story using the following information: {question}"},
        {"role": "assistant", "content": "Story: "}
    ], # TODO: Follow the n_retrival question format.,
    "IDK":[
        {"role": "system", "content": "Your task is to answer the user’s question based on a long context, which consists of many bios. Output the answer only. If you don't know the answer or the answer is no explicitly stated, you should strictly output 'The answer is not stated.'. Don't explain or output other things."},
        {"role": "user", "content": "Context: {given_context}\n\nQuestion: {question} Output 'The answer is not stated' if the answer is not explicitly told in the context."},
        {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
    ],
    "IDKForBase":"Context: {given_context}\n\nQuestion: {question} \nAnswer the question. If you don't know the answer or the answer is no explicitly stated, you should strictly output 'The answer is not stated.'.\n\nAnswer:",
}

CITE_PROMPT_TEMPLATE = {
    "SingleQA": [
    {"role": "system", "content": "Your task is to answer the user’s question with citation based on a long context, which consists of many bios. You must output the answer following with the citation number of the relevant bios strictly surrounded by square brackets such as [1]. Don't explain or output other things."},
    {"role": "user", "content": "Context: {given_context}\n\nQuestion: {question}\nAnswer the question followed by the reference bio number as citation."},
    {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
    ],
    "MultiQA": [
    {"role": "system", "content": "Your task is to answer all the user’s questions with citation based on a long context, which consists of many bios. You must output the answer following with the citation number of the relevant bios strictly surrounded by square brackets such as [1]. You should give the answer and citation for each question sequentially. Don't explain or output other things."},
    {"role": "user", "content": "Context: {given_context}\n\nThe Questions are as follows:\n{question}\nAnswer each question followed by the reference bio number citation in sequence."},
    {"role": "assistant", "content": "Based on the provided context, the answer to each question is as follows:"}
    ],
    "SingleFewShotQA": [
    {"role": "system", "content": "Your task is to answer the user’s question with citation based on a long context, which consists of many bios. Following the format of the examples, you must output the answer ending with the citation number of the relevant bios strictly surrounded by square brackets such as [1]. Don't explain or output other things."},
    {"role": "user", "content": "Context: {given_context}\n\n{examples}\n\nQuestion: {question}\nAnswer: "},
    {"role": "assistant", "content": "Based on the provided context, {question_prefix}"}
    ],
    "MultiFewShotQA": [
    {"role": "system", "content": "Your task is to answer all the user’s questions with citation based on a long context, which consists of many bios. Following the format of the examples, You must output the answer ending with the citation number of the relevant bios strictly surrounded by square brackets such as [1]. You should give the answer and citation for each question sequentially. Don't explain or output other things."},
    {"role": "user", "content": "Context: {given_context}\n\n{examples}\n\n{question}\nAnswers:\n"},
    {"role": "assistant", "content": "Based on the provided context,"}
    ],
    "SingleFewShotQAForBase": "Context: {given_context}\n\n{examples}\n\nQuestion: {question} \nAnswer: Based on the provided context, {question_prefix}",
    "MultiFewShotQAForBase": "Context: {given_context}\n\n{examples}\n\n{question}\nAnswers:\nBased on the provided context,"
}


IDK_PROMPT_TEMPLATE = [
    {"role": "system", "content": "Your task is to answer the user’s question based on a long context, which consists of many bios. Output the answer only. If you don't know the answer or the answer is no explicitly stated, you should strictly output 'The answer is not explicitly stated.'. Don't explain or output other things."},
    {"role": "user", "content": "Context: {given_context}\n\nQuestion: {question}"},
    {"role": "assistant", "content": "Based on the provided context, {question_prefix} "}
]