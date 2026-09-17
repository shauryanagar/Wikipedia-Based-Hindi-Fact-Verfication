import gradio as gr
from pipeline import verify_claim

def check_claim(claim_text):
    if not claim_text or not claim_text.strip():
        return "N/A", "0.0%", "N/A", "Please enter a valid claim in Hindi."
    
    result = verify_claim(claim_text)
    
    verdict = result.get("verdict", "NOT ENOUGH INFO")
    confidence = result.get("confidence", "0.0%")
    sources = ", ".join(result.get("sources", [])) if result.get("sources") else "No sources found"
    evidence = result.get("evidence", ["No evidence extracted."])[0]
    
    return verdict, confidence, sources, evidence

examples = [
    ["नरेंद्र मोदी भारत के वर्तमान प्रधानमंत्री हैं।"],
    ["महात्मा गांधी का जन्म 1950 में हुआ था।"],
    ["चंद्रमा पृथ्वी का एकमात्र प्राकृतिक उपग्रह है।"],
    ["भारत की राजधानी मुंबई है।"],
    ["नई दिल्ली भारत की आधिकारिक राजधानी है।"]
]

with gr.Blocks(title="Hindi Fact Verification System") as demo:
    gr.Markdown("# Wikipedia-Based Hindi Fact Verification System")
    gr.Markdown("Automated claim verification pipeline using Hindi Wikipedia, MuRIL, Hindi SBERT, and mDeBERTa-v3.")

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Input Claim (Hindi)",
                placeholder="Enter a claim or statement in Hindi...",
                lines=3
            )
            submit_btn = gr.Button("Verify Claim", variant="primary")
            gr.Examples(examples=examples, inputs=input_text)
        
        with gr.Column(scale=2):
            output_verdict = gr.Textbox(label="Verdict")
            output_confidence = gr.Textbox(label="Confidence")
            output_sources = gr.Textbox(label="Referenced Wikipedia Sources")
            output_evidence = gr.Textbox(label="Extracted Evidence (Premise)", lines=4)

    submit_btn.click(
        fn=check_claim,
        inputs=input_text,
        outputs=[output_verdict, output_confidence, output_sources, output_evidence]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)