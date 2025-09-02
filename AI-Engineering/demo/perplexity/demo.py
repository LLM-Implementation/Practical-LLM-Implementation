# First, install the necessary library
# pip install evaluate

# 1. Import the perplexity metric
from evaluate import load
perplexity_metric = load("perplexity")

# 2. Choose a small model
model_id = "distilgpt2" # Fast and free to run

# 3. Define our test texts
html_text = """
<!DOCTYPE html>
<html>
<head>
<title>Page Title</title>
</head>
<body>
<h1>This is a Heading</h1>
<p>This is a paragraph.</p>
</body>
</html>
"""

prose_text = """
In the heart of the whispering woods, where time itself seemed to meander like a lazy river, a lone cartographer charted the celestial ballet of forgotten constellations, his ink a concoction of starlight and memory, each stroke a testament to the silent, swirling chaos of the cosmos.
"""



# 4. Calculate perplexity for HTML (structured, predictable text)
perplexity_metric.add_batch(predictions=[html_text])
html_result = perplexity_metric.compute(model_id=model_id)
html_ppl = html_result['mean_perplexity']

# 5. Calculate perplexity for prose (creative, less predictable text)
perplexity_metric.add_batch(predictions=[prose_text])
prose_result = perplexity_metric.compute(model_id=model_id)
prose_ppl = prose_result['mean_perplexity']

# 6. Display beautiful results
print("🚀 ===== AI Evaluation Demo: Perplexity =====")
print("📖 Understanding Perplexity:")
print("   Perplexity measures how 'surprised' a language model is by text.")
print("   • Lower score = More predictable text")
print("   • Higher score = More surprising/unpredictable text")
print("   ──────────────────────────────────────────────────────────")

print(f"\n📊 RESULT - Structured HTML:")
print(f"   🎯 Perplexity Score: {html_ppl:.2f}")
print("   💡 Interpretation: Low score indicates predictable, structured text")
print("   📝 Expected for: Code, markup, formal documents")

print(f"\n📊 RESULT - Unstructured Prose:")
print(f"   🎯 Perplexity Score: {prose_ppl:.2f}")
print("   💡 Interpretation: Higher score indicates creative, unpredictable text")
print("   📝 Expected for: Poetry, creative writing, complex narratives")

# Comparison section
print(f"\n🔍 COMPARISON:")
diff = prose_ppl - html_ppl
ratio = prose_ppl / html_ppl
print(f"   📈 Difference: {diff:.2f} points")
print(f"   🔄 Ratio: {ratio:.1f}x (prose is {ratio:.1f} times more unpredictable)")
print("   ✅ This demonstrates how perplexity quantifies text complexity!")

print("\n🎉 Demo completed! Perplexity helps evaluate text predictability and complexity.")
print("   Try modifying the text samples to see how different content affects the scores!")
