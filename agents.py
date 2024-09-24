class AgentProfile:
    def __init__(self, profile: dict):
        self.profile_dict = profile
        # first_name: str
        # last_name: str
        # age: int
        # occupation: str
        # gender: str
        # gender_pronoun: str 
        # public_info: str
        # big_five: str
        # moral_values: list[str]
        # schwartz_personal_values: list[str]
        # personality_and_values: str
        # decision_making_style: str
        # secret: str
    

json_schema_agents = {
    "type": "object",
    "properties": {
        "characters" : {"type" : "array",
        "items" : {"type" : "object",
        "properties" : {
            "first_name" : {"type" : "string"},
            "last_name" : {"type" : "string"},
            "age" : {"type" : "number"},
            "occupation": {"type" : "string"},
            "gender" : {"type" : "string"},
            "gender_pronoun" : {"type" : "string"}, 
            "public_info" : {"type" : "string"},
            "big_five" : {"type" : "string"},
            "moral_values"  : {"type" : "string"},
            "schwartz_personal_values": {"type" : "array", "items" : {"type": "string"}},
            "decision_making_style": {"type" : "string"},
            "secret": {"type" : "string"},
            "social_goal" : {"type" : "string"}
        } }
        }
    },
    "required": ["description", "goal", "constraints", "interaction"]
}

def llama_3_extract_agents(movie_scene):
    print("Extracting agents")
    json_format = {
        "characters": [{
        "first_name" : "First name of the character",
        "last_name" : "Last name of the character",
        "age" : "Age of the character",
        "occupation": "Job/Occupation of the character",
        "gender" : "gender",
        "gender_pronoun" : "Pronouns of the character", 
        "public_info" : "Piece of public information that other people know about them",
        "big_five" : "One of the 'Big Five Markers' or 'Big Five personality traits', which are \
        used to describe human personality. Can be  “openness to experience”, “conscientiousness”, “extraversion”,“agreeableness” and “neuroticism”",
        "moral_values"  : "Type of moral values. The moral value types are “care”, “fairness”, “loyalty”, “authority” and “purity”",
        "schwartz_personal_values": ["One or more of the ten broad personal values that fits the character. The values come\
        from schwartz theory of basic human values. Can be “self-direction”, “simulation”, “hedonism”, “achievement”, “power”, “security”, “conformity”,“tradition”, “benevolence”, and “universalism”"],
        "decision_making_style": "The decision-making style types are “directive”, “analytical”, “conceptual”, and “behavioral”",
        "secret": "a secret that this character doesn not want anyone else to know", 
        "social_goal" : "The goal that the character is trying to achieve in the current movie scene"
        }] 
    }

    system_message = f"""
    #### Persona: ###
    You are an expert on movie characters. You know every detail about any character participating in arbitrary movie scene.
    On top of knowing character's name, personality traits, etc. in the presented movie scene, you also know their motivations, secrets and social goals that they are trying to achieve
    ### Goal: ###
    You will be given a movie scene. You will have to extract information about the characters present in that movie scene. There can be multiple characters. The information that you'll extract
    includes:
    - First name
    - Last name
    - Age of the character,
    - Job/Occupation of the character,
    - Gender,
    - Pronouns of the character, 
    - Piece of public information that other people know about them,
    - One of the 'Big Five Markers' or 'Big Five personality traits', which are \
    used to describe human personality. Can be  “openness to experience”, “conscientiousness”, “extraversion”,“agreeableness” and “neuroticism”",
    - Type(s) of moral values. The moral value types are “care”, “fairness”, “loyalty”, “authority” and “purity”. Can be more than one,
    - One or more of the ten broad personal values that fits the character. The values come\
    from schwartz theory of basic human values. Can be “self-direction”, “simulation”, “hedonism”, “achievement”, “power”, “security”, “conformity”,“tradition”, “benevolence”, and “universalism”,
    - The decision-making style that fits the character.Styles are “directive”, “analytical”, “conceptual”, and “behavioral”,
    - A secret that this character doesn not want anyone else to know
    - The social goal that the character is trying to achieve in this scene
    """

    # TODO make a good prompt here 
    user_message = f""" 
    ### Question: ###
    Please find a movie scene that demonstrates the {movie_scene} type of human social interaction. Your output should be a description of a movie
    scene with accordance to your instructions.

    ### Format: ###
    Use the following json format:
    {json_format}
    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    return messages

def extract_movie_scene(movie_scene: dict, model, tokenizer, json_schema, max_length=500):
    messages = llama_3_extract_agents(movie_scene)
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt,max_number_tokens=max_length,
                        max_array_length=max_length,
                        max_string_token_length=max_length)
    extracted_data = jsonformer()
    return extracted_data

def init_instruction(scenario, movie_character, movie_info, start=True):

        json_format = {
        "init_msg" : "set to true if it is an initial message. set to false, when the agent is supposed to start a conversation",
        "utterance" : "Message conveyed by the agent. If it is an initial message, then it is empty"
    }

    if start:
        init_msg = "Please start the conversation"
    else:
        init_msg = "You do NOT start the conversation. You will wait for the first message from another agent"

    system_message = f"""
    #### Persona: ###
    You will engage in conversation which unfolds in the given scenario. The main idea is that, while playing the scenario out, you will act like 
    a given movie character from a given movie. You will know all the details about your assigned movie character including: motivations, 
    personality traits, personal values, morality type etc. 
    ### Goal: ###
    Successfully conduct a conversation reaching your goal.
    """

    # TODO make a good prompt here 
    user_message = f""" 
    ### Question: ###
    You will act like movie character {movie_character} from the {movie_info["movie_name"]} movie. Your goal is to achieve {movie_info["goal"]} 
    in the following scenario: {scenario}. You will now engage in conversation with another agent. {init_msg}


    ### Format: ###
    Use the following json format:
    {json_format}
    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    return messages

def return_utterance(previous_utterance):

    json_format = {
        "previous_utterance" : "set to true if it is an initial message. set to false, when the agent is supposed to start a conversation",
        "utterance" : "Message conveyed by the agent. If it is an initial message, then it is empty"
    }

    # TODO make a good prompt here 
    user_message = f""" 
    ### Question: ###
    Following the previous conversation and previous utterance from your conversation partner: {previous_utterance}.
    Respond accordingly.


    ### Format: ###
    Use the following json format:
    {json_format}
    """
    messages = [
            {"role": "assistant", "content": user_message},
        ]
    return messages

def run_turn(previous_utterance):
    json_schema = {
    "type": "object",
    "properties": {
        "previous_utterance" : {"type" : "string"}
        "utterance" : {"type" : "string"}
    }
    }
    messages = return_utterance(previous_utterance)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt,max_number_tokens=max_length,
                        max_array_length=max_length,
                        max_string_token_length=max_length)
    extracted_data = jsonformer()

def init_conversation(scenario, movie_character, movie_info, start):
    json_schema = {
    "type": "object",
    "properties": {
        "init_msg" : {"type" : "boolean"}
        "utterance" : {"type" : "string"}
    }
    }
    messages = init_instruction(scenario, movie_character, movie_info, start=True)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt,max_number_tokens=max_length,
                        max_array_length=max_length,
                        max_string_token_length=max_length)
    extracted_data = jsonformer()
    return extracted_data

class Conversation:
    def __init__(self, scenario, characters, movie_info, MODEL_NAME):
        self.model_name = MODEL_NAME
        self.agent_1_name, self.agent_2_name = characters # assume 2 characters
        self.agent_1 = self._init_model()
        self.agent_2 = self._init_model()
        self.tokenizer = tokenizer
        self.scenario = scenario
        self.episode = []

    def _init_agents(self):
        _ = init_conversation(scenario, self.agent_2, movie_info, start=False)
        first_utterance["utterance"] = init_conversation(scenario, self.agent_1, movie_info, start=True)
        self.episode.append(first_utterance)
    def _init_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = 'auto')
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        return model
    def run_conversation(self):
        while len(episode) < 20:
            if len(episode) == 1:
                agent_2_utter = run_turn(self.episode[-1])
                self.episode.append(agent_2_utter)
            else:
                agent_1_utter = run_turn(self.episode[-1])
                agent_2_utter = run_turn(agent_1_utter)
                self.episode.append(agent_1_utter)
                self.episode.append(agent_2_utter)
    def return_episode(self):
        return self.episode






