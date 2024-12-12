import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import os
from tqdm import tqdm
import logging
import copy

class MistralPrompt:
    def __init__(
        self, 
        post_title, 
        post_content, 
        user_comment_history, 
        previous_comments_on_thread, 
        comment_of_interest,    
        ): 
        self.post_title = post_title
        self.post_content = post_content
        self.user_comment_history = user_comment_history
        self.previous_comments_on_thread = previous_comments_on_thread
        self.comment_of_interest = comment_of_interest

    def benchmark_prompt(self):
        if self.comment_of_interest == []: 
            self.comment_of_interest == "".join(self.post_title, self.post_content)

        mistral_prompt = f"""
        [INST] 
        You are mimicking a real reddit user. Your task is to respond to the comment of interest. Your response should be in english and not include any explanation or notes about your response:
        Comment of interest: {self.comment_of_interest}
        <<ANSWER>>
        [/INST]
        """
        return mistral_prompt

    def zero_shot_prompt(self):
        mistral_prompt = f"""
        [INST] 
        You are mimicking a real reddit user. Your task is to respond to the comment of interest based on the post title and post information, the comments in the thread, and the user's comment history that you are mimicking. If there is no comment of interest, respond to the post.  Your response should be in english, mimic the reddit user, and not include any explanation or notes about your response:
        The post's title: {self.post_title}
        The post's content: {self.post_content}
        User's comment history: {self.user_comment_history}
        Previous user comments on the thread: {self.previous_comments_on_thread}
        Comment of interest: {self.comment_of_interest}
        <<ANSWER>>
        [/INST]
        """
        return mistral_prompt


    def few_shot_prompt(self):
        
        mistral_prompt = f"""
        </s>
        [INST] 
        You are mimicking a real reddit user. Your task is to respond to the comment of interest based on the post title and post information,  the comments in the thread, and the user's comment history that you are mimicking. If there is no comment of interest, respond to the post. Your response should be in english, mimic the reddit user, and not include any explanation or notes about your response:
        For instance, the following:
        The post's title: What do I do with old books?
        The post's content: So I’ve got 6 shelves worth of children’s books that are aimed at 8-13y kids that I need to get rid off to make space for my current books and revision resources. I’ve tried to sell them on eBay, but it really didn’t work and since a lot of them are singular and not collection, no one really wanted them. I don’t want them to got to waste since they are in great condition, I’m considering charity as my last resort as I’ve been told they throw away many books if they don’t sell. I don’t mind not making any money from it but I really do want them to be appreciated by people.
        User's comment history: ['No, meaning the heros blame themselves for things they didn't do and have unnecessary guilt.', 'YESSSS!! already upvoted, but wanted to emphasize it more lol absolutely!! And when they think something horrible that happened is their fault when it isn't... Sarah J Maas's books have some of the most self-hating characters I've ever seen! It makes me roll my eyes.', 'I wasn't too crazy over Fourth Wing, DNFd it, and the only reason I keep thinking about it is because it pops up everywhere lol', 'I can\'t stand when authors use 'make no mistake'!', 'When the cover design changes in the middle of a series to force more sales so people rebuy the first books *cough* ACOTAR *cough*', '👏🏻👏🏻', 'Storygraph is better than Goodreads in my opinion as tracker that matches you with books based on what you've read already. It has in-depth stats too that Goodreads doesn't have. Highly recommend!', 'definitely ACOTAR and ToG by SJM', 'I think you just haven't found the right person. Reading romance books will only give you unrealistic expectations about what romance should be. You'll find the right person, don't worry. 🥰', 'I completely agree cracked spines look more loved! 🥰 my biggest ick with other readers is when they fight to the death about how amazing or how horrible a book is when it's their own opinion, and everyone is entitled to their own. I've had people actually stop talking to me after mentioning a book I didn't like because it turned out to be their favorite 🙄 people have to get over themselves. If everyone agreed, the world would be a boring place, and every book would be the same!']
        Previous user comments on the thread: 
        Comment of interest:
        would be responded to with [/INST] the best places are schools, who are always underfunded and need more books, and libaries. the libraries sell them during quarterly book sales to make extra money.

        For instance, the following:
        The post's title: How does Harry Potter work for kids nowadays?
        The post's content: One of the things I've always found the most interesting about the *Harry Potter* series (and I think one of the reasons it's had so much success) is that each successive book feels like it's aimed at a slightly higher age group. *Harry Potter and the Philosopher's Stone* is definitely a children's book, and I wouldn't say that *Harry Potter and the Deathly Hallows* is. For the original generation of readers, this worked out well because we grew up with the characters.
        User's comment history: 
        Previous user comments on the thread: 
        Comment of interest: 
        would be responded to with [/INST] I was 11 years old when I read the first book (and I promptly read the rest of them within that week). I did reread the series several times, and had a different perspective watch time. I’d say 10+ and they should be okay reading the full series. Definitely depends on their reading level and maturity in how much they understand, but you’d be surprised at how much they can grasp/understand at that age - and if there’s anything too much it usually goes over their head until they come back for a reread. My thought’s definitely changed on how I view the characters and story over time, but that’s part of the process so I don’t really think there’s a perfect time. For example, for anyone that’s older and holding off on reading Harry Potter because it’s a “children’s book” - I’d say you’re at the perfect age to go ahead and read it.

        For instance, the following:
        The post's title: What’s a book that you feel was marketed incorrectly?
        The post's content: By this I basically mean, a book that was marketed as one thing and either didn’t live up to what was promised or was something else entirely. I feel this is very common practise especially nowadays due to the rise of TikTok leading a lot of books to be simplified down to their tropes. A book I feel was marketed incorrectly is probably Song Of Achillies by Madeline Miller. I read the UK edition of this book, and on the cover the book is described as being ‘dark’ and ‘sexy’. After having read the book this is pretty disturbing given that the only sexual scenes take place when the characters are very young / underage (all others fade to black). Not to mention one of the scenes is purposefully very uncomfortable. Maybe that’s a desperate discussion for the sexualisation of queer relationships, it just especially disturbed me with TOSA. It’s a very beautiful book by all means, but why is it marketed as being ‘sexy’ when it really isn’t that at all, though maybe it’s just marketed that way in the UK. What books to you feel weren’t marketed correctly?
        User's comment history: ['A few reasons. Firstly, it's rare to find it written well. I've actually had better luck with fanfics and fanfic-style shorts online when I'm looking for smut. The balance between plot and sex in a full length novel is much more difficult. Which brings me on to the next reason, which is that when I'm reading a book it's normally because I want to enjoy a story, not because I want to get off on it. This means if an explicit sex scene does come up, it does nothing for me and I'm usually just bored and want to get back to the story. Then there's how it's rarely ever relevant to the actual story being told. The fact that the characters have sex may be relevant, but the details of what happens feel pointless. Hence why I much prefer it when books show the lead up, which is where the important character moments typically are, then end the scene as it begins. It's not ignored, it just doesn't waste time describing things not relevant to the story being told.', 'Unfortunately I have no idea. It is quite possibly a case of certain people with negative views on sex in general lashing out. I'd guess that most of these types of people know on some level that it's unacceptable, which is why they tend to downvote instead of reply. I think you're mistaken when you assume that this subreddit should be more rational and open minded. From what I've seen of this place, there's often an underlying tone of hostility, and a sense of superiority based on what kinds of books they enjoy. I don't know why, but it's the reason I don't often comment in this sub.', 'I finished it, and it does not get better. The writing gets worse as soon as the love interest becomes a love interest, with half the internal dialogue just being descriptions of the ways she's attracted to him and struggling to resist him. It made me feel like I was reading the diary of a 14 year old, when the main character is apparently a 20 year old. I also really wanted to like it because of the dragons, but they're barely in it. When they finally show up almost half way through, I was hopeful it was going to get better, as I was actually enjoying their interactions at first. But then they're just not present for most of it, and even though they have the ability to talk to each other at any time, present or not, there's rarely more than a single sentence spoken to each other. They are not developed at all.']
        Previous user comments on the thread: ['Fourth wing. It was marketed as the new game of thrones and to be more violent and messy than anything seen before. But it was super predictable pseudo-enemies to lovers that was confusingly world built', 'Ok I just started this and I do enjoy it but I can tell it’s predictable lol does the main character fall in love with who I think she falls for? >!xander!<', 'I’m so sorry you had to see my comment, but yes. I didn’t mean to spoil it for you', 'I really love spoilers actually and tbh I had guessed it from her first description when she saw him.', 'It really doesn’t try to be subtle lol. And I hope you enjoy the series, it’s just not my thing', 'I think it might just be a good transition to adult books. I don’t really read many adult books but YA is so meh to me now.', ">I think it might just be a good transition to adult books. It's... really not."]
        Comment of interest: 'If I enjoy it what’s the problem? I don’t read a lot of adult books so if I find something I like and read it and that gets me to find more adult books then it’s worked.
        would be responded to with [/INST] It depends what exactly you mean by ""adult"". If you mean a more mature book, then definitely not. Despite the main characters supposedly being 20, the way it's written makes them sound like mid-teens, with pretty much no emotional maturity to be found. It's also a relatively simple world/plot, and with only a single viewpoint character there's not much to follow. I've read plenty of ""YA"" that is more mature/complex. If you mean the explicit sex scenes, then sure. They didn't really do anything for me, but clearly it does for others, so if you want a starting point for those types of books then it could work. From what I remember it was rather vanilla in that aspect.
        </s>
        
        [INST]
        The post's title: {self.post_title}
        The post's content: {self.post_content}
        User's comment history: {self.user_comment_history}
        Previous user comments on the thread: {self.previous_comments_on_thread}
        Comment of interest: {self.comment_of_interest}
        <<ANSWER>>
        [/INST]
        """
        return mistral_prompt

class Mistral:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.load_quantized_model()
        self.tokenizer = self.initialize_tokenizer()

        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            # If an eos_token exists, reuse it as pad_token
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Otherwise, add a new PAD token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))

    def load_quantized_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        return model

    def initialize_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.bos_token_id = 1
        return tokenizer

    def run_prompt_on_mistral(self, prompts):
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize all prompts together
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,    
            truncation=True   
        )

        encoded = encoded.to("cuda")

        generated_ids = self.model.generate(
            **encoded,
            max_new_tokens=200,
            do_sample=True,
            num_return_sequences=1 
        )

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return decoded



    def process_post_tree(self, post_tree, user_histories, data_basepath, test_type):
        if test_type not in ['benchmark', 'zero_shot', 'few_shot', 'instruct']:
            raise ValueError(f"Invalid test_type provided: {test_type}. Valid types: zero_shot, few_shot, instruct")

        new_post_tree_file = os.path.abspath(
            os.path.join(data_basepath, f'posttrees/{test_type}/{post_tree.post_id}.json')
        )

        post_tree_copies = {f'copy_{i}': copy.deepcopy(post_tree) for i in range(10)}

        original_nodes = list(post_tree.bfs_generator())
        copies_nodes = {
            name: list(tree_copy.bfs_generator()) 
            for name, tree_copy in post_tree_copies.items()
        }

        total_nodes = len(original_nodes)

        with tqdm(total=total_nodes, desc="Generating responses", unit="node") as pbar:
            for idx, original_node in enumerate(original_nodes):
                batched_prompts = []
                for copy_name, tree_copy in post_tree_copies.items():
                    copy_node = copies_nodes[copy_name][idx]
                    
                    cur_user_comment_history = user_histories.get_random_user_history(f"{original_node.user}")

                    previous_comment_tree = original_node.get_previous_responses()
                    previous_comments = [c.comment_text for c in previous_comment_tree] if previous_comment_tree else []

                    if previous_comments:
                        comment_of_interest = previous_comments.pop()
                    else:
                        comment_of_interest = ""

                    mistral_prompt = MistralPrompt(
                        post_title=post_tree.title,
                        post_content=post_tree.content,
                        previous_comments_on_thread=previous_comments,
                        comment_of_interest=comment_of_interest,
                        user_comment_history=cur_user_comment_history,
                    )

                    if test_type == "zero_shot":
                        prompt = mistral_prompt.zero_shot_prompt()
                    elif test_type == "few_shot":
                        prompt = mistral_prompt.few_shot_prompt()
                    elif test_type == "benchmark": 
                        prompt = mistral_prompt.benchmark_prompt()
                    else:
                        raise ValueError("Invalid test_type or not implemented")

                    batched_prompts.append(prompt)

                # run all 10 prompts together in one batch call
                all_responses = self.run_prompt_on_mistral(batched_prompts)

                for (copy_name, tree_copy), response in zip(post_tree_copies.items(), all_responses):
                    if '<<ANSWER>>' in response:
                        filtered_response = response.split('<<ANSWER>>', 1)[-1].strip()
                    else:
                        filtered_response = response.strip()

                    copy_node = copies_nodes[copy_name][idx]
                    copy_node.comment_text = filtered_response

                pbar.update(1)

        for i, (copy_name, tree_copy) in enumerate(post_tree_copies.items()):
            copy_file = os.path.abspath(os.path.join(data_basepath, f'posttrees/{test_type}/{post_tree.post_id}/copy_{i}.json'))
            tree_copy.save_as_json(copy_file)
            logging.info(f"Processing of {copy_name} completed and saved to {copy_file}")
        
        post_tree.save_as_json(os.path.abspath(os.path.join(data_basepath, f'posttrees/{test_type}/{post_tree.post_id}/original.json')))