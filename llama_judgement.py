from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def judge_answer(prompt, provided_answer):
    evaluation_prompt = f"""
    Given the following prompt and answer, judge whether the answer is correct or not:

    Prompt: {prompt}
    Answer: {provided_answer}

    Is the answer correct? Answer with "Yes" or "No".
    """
    
    inputs = tokenizer(evaluation_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if "Yes" in result:
        return True
    elif "No" in result:
        return False
    else:
        print("Model response unclear:", result)
        return None

def main(prompt, provided_answer):
    print("Prompt:", prompt)
    print("Provided Answer:", provided_answer)
    
    # Judge the provided answer
    is_correct = judge_answer(prompt, provided_answer)
    if is_correct is True:
        print("The provided answer is correct.")
    elif is_correct is False:
        print("The provided answer is incorrect.")
    else:
        print("The model's response was unclear.")

if __name__ == "__main__":

    prompt = """ human social interaction types are from following categories: exchange, competition, cooperation, conflict, and coercion.  Please provide a detailed description of a movie scene that demonstrates the "exchange" type of human social interaction.
    Include the following details:
    - A description of the scene
    - The characters involved
    - The setting
    - The nature of the conflict or interaction 

    Based on the identified movie scene, please generate a scenario of human interactions and provide a detailed description of it. 
    When generating scenarios, substitute the names of movie characters with more general terms such as "agents" to ensure broader applicability.
    Include the following details:

    1. A description of the scene
    2.Goal: Describe the goal of the human interaction scenario.
    3. Constraints: List the constraints or conditions related to the scenario.
    4. Interaction: Describe the detailed interaction between the agents in the scenario.
    """
    provided_answer = """
    Title: The Trade

    Scene Description:
    The scene unfolds in a bustling city marketplace during the late afternoon. Sunlight filters through the gaps between colorful awnings, casting a warm glow on the cobblestone streets. The camera pans across various stalls, each offering an array of goods: fruits, textiles, and trinkets. The market is lively with vendors calling out their wares and customers haggling over prices.

    Characters Involved:

    Sarah: A middle-aged woman who owns a stall selling handmade jewelry. She’s known for her exquisite designs and is well-regarded in the market for her fair trade practices.
    Tom: A young man who runs a small stall selling fresh produce. His stand is modest, but he takes pride in offering high-quality fruits and vegetables.
    Mrs. Thompson: An elderly lady and regular customer, known for her frequent visits to both Sarah and Tom's stalls.
    Setting:
    The scene is set in a vibrant and bustling outdoor market. Sarah’s jewelry stall is adorned with intricate necklaces and bracelets, while Tom’s stall is filled with colorful fruits and vegetables. The market is surrounded by a mix of historic and modern buildings, giving it a charming yet dynamic atmosphere.

    Nature of the Conflict or Interaction:
    In this scene, Sarah and Tom are engaged in a friendly yet purposeful exchange. Sarah approaches Tom’s stall with a small basket of her jewelry, and Tom greets her with a warm smile. They are negotiating a trade: Sarah needs some fresh produce for an upcoming event she’s hosting, and Tom is interested in acquiring a few pieces of Sarah’s jewelry as gifts for his family.

    The interaction is characterized by mutual respect and the desire to find a fair trade. Sarah examines Tom’s selection of fruits, choosing a few apples and oranges while Tom inspects the jewelry, selecting a bracelet and a pair of earrings. They discuss the value of each item, weighing the worth of the produce against the jewelry. They both agree on a fair exchange rate, with Tom offering a bit more produce than originally planned as a gesture of goodwill and Sarah reciprocating by adding a small extra piece of jewelry.

    Mrs. Thompson, who is browsing nearby, observes the interaction with interest. She eventually joins in, buying a piece of jewelry from Sarah and a basket of fresh produce from Tom. This spontaneous purchase further underscores the successful exchange and reinforces the sense of community within the market.

    Summary:
    The scene exemplifies the "exchange" type of human social interaction through the mutual and voluntary trade of goods between Sarah and Tom. Their interaction highlights the principles of fair trade and cooperation, where both parties benefit from the transaction and maintain a positive relationship. The lively market setting enhances the ambiance, illustrating how exchanges can occur in everyday social environments, fostering both economic and social connections.                                                                      Scene Description:
    The scene takes place in a vibrant outdoor marketplace, set in a historic district of a bustling city. The market is filled with colorful stalls and booths, each offering a variety of goods such as fresh produce, handmade crafts, and local specialties. The atmosphere is lively with the hum of conversation, the aroma of fresh foods, and the clinking of goods being exchanged. In the midst of this activity, two distinct stalls catch the eye: one is adorned with beautifully crafted jewelry, while the other is piled high with fresh fruits and vegetables.

    Goal:
    The goal of the human interaction scenario is to facilitate a fair trade between two vendors—one selling jewelry and the other selling fresh produce. Each vendor seeks to obtain goods they need by exchanging their own products. The jewelry vendor aims to acquire fresh produce for personal use or resale, while the produce vendor wants to obtain a piece of jewelry, possibly as a gift or for personal enjoyment.

    Constraints:

    Value Agreement: Both vendors must agree on a fair exchange value that reflects the worth of their respective goods.
    Product Availability: The jewelry vendor needs specific types of produce, and the produce vendor is interested in particular jewelry pieces, which may limit options.
    Market Time: The exchange must be completed within the operating hours of the market, which are constrained by the end of the business day.
    Quality Assurance: Both vendors need to ensure that the goods being exchanged are of satisfactory quality and meet each other’s expectations.
    Interaction:

    Initial Approach: The jewelry vendor approaches the produce vendor’s stall with a small selection of jewelry pieces, including bracelets and necklaces. The produce vendor greets them with a friendly smile, recognizing them from previous visits.

    Presentation of Goods: The jewelry vendor begins by displaying a few pieces of jewelry, explaining their craftsmanship and highlighting the quality of materials used. The produce vendor examines the jewelry with interest and selects a couple of pieces that catch their eye.

    Assessment and Negotiation: The produce vendor presents a variety of fresh fruits and vegetables, describing their freshness and origin. Both parties discuss their respective needs and preferences— the jewelry vendor specifies their requirements for certain types of produce, while the produce vendor indicates their interest in particular jewelry items.

    Valuation Discussion: The two vendors engage in a negotiation to determine a fair trade. They compare the value of the jewelry pieces to the produce, taking into account factors like the cost of materials, market prices, and personal preferences. They discuss and agree on a quantity of produce that matches the value of the selected jewelry items.

    Final Agreement: After some back-and-forth negotiation, they reach a consensus on the exchange terms. The jewelry vendor agrees to provide a specific piece of jewelry in exchange for a selection of fresh produce. They shake hands, solidifying the deal, and both parties express satisfaction with the arrangement.

    Transaction Completion: The jewelry vendor leaves a piece of jewelry with the produce vendor, and in return, receives a basket of fresh fruits and vegetables. The transaction is completed with a friendly exchange of pleasantries, and both vendors are pleased with the outcome of the trade.

    Additional Purchases: Observing the interaction, another market-goer approaches and buys a piece of jewelry from the vendor and a few items of produce from the produce vendor, further supporting the notion of fair exchange and enhancing the sense of community within the market.

    Summary:
    In this scenario, the human interaction focuses on a mutual and voluntary exchange between two vendors. Each agent aims to obtain goods they need through a fair trade, navigating constraints related to value, availability, and market conditions. The detailed interaction showcases negotiation, product evaluation, and agreement, reflecting the principles of the exchange type of social interaction.
    """

    main(prompt, provided_answer)
