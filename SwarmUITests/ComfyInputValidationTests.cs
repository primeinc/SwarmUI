using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SwarmUI.Builtin_ComfyUIBackend;

namespace SwarmUITests;

/// <summary>Tests <see cref="ComfyInputValidation"/> against ComfyUI 7ef81780: each branch of validation.py validate_node_input, the upstream cases in tests-unit/execution_test/validate_node_input_test.py,
/// the dynamic-input expansions of comfy_api/latest/_io.py with cases from tests-unit/execution_test/dynamic_inputs_test.py and tests-unit/comfy_api_test/io_dynamic_group_test.py, and the link branch of execution.py validate_inputs.</summary>
[TestFixture]
public class ComfyInputValidationTests : SwarmUITest
{
    /// <summary>Prepares the basics.</summary>
    [OneTimeSetUp]
    public static void PreInit()
    {
        Setup();
    }

    /// <summary>Characters Python's str.isspace accepts, observed with the ComfyUI venv's Python 3.13.15: <c>[hex(i) for i in range(0x110000) if chr(i).isspace()]</c>. None are above U+FFFF.</summary>
    public static readonly int[] PythonWhitespace = [0x9, 0xa, 0xb, 0xc, 0xd, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000];

    /// <summary>The whitespace str.strip removes is exactly Python's set, which is .NET's plus U+001C to U+001F.</summary>
    [Test]
    public void TestPythonWhitespaceMatchesPython()
    {
        int[] ported = [.. Enumerable.Range(0, 0x10000).Where(c => ComfyInputValidation.IsPythonWhitespace((char)c))];
        Assert.That(ported, Is.EqualTo(PythonWhitespace));
        int[] dotnet = [.. Enumerable.Range(0, 0x10000).Where(c => char.IsWhiteSpace((char)c))];
        Assert.That(PythonWhitespace.Except(dotnet), Is.EqualTo(new[] { 0x1c, 0x1d, 0x1e, 0x1f }), "the term .NET lacks");
        Assert.That(ComfyInputValidation.PythonStrip("\u001c IMAGE\u001f\t"), Is.EqualTo("IMAGE"));
        Assert.That(ComfyInputValidation.PythonStrip("  "), Is.EqualTo(""));
    }

    /// <summary>One validate_node_input case: received and input types as JSON, the expected result, and the validation.py line that decides it.</summary>
    public record LinkTypeCase(string Received, string Input, bool Accepted, string Branch);

    /// <summary>Cases for every branch of validation.py validate_node_input with strict=False, in branch order.</summary>
    public static readonly LinkTypeCase[] BranchCases =
    [
        new("\"STRING\"", "\"STRING\"", true, "213 equal strings"),
        new("[\"a\",\"b\"]", "[\"a\",\"b\"]", true, "213 equal choice lists"),
        new("[1,2]", "[1.0,2.0]", true, "213 Python compares numbers by value"),
        new("[true]", "[1]", true, "213 a Python bool equals 0 or 1"),
        new("[\"a\",\"b\"]", "[\"b\",\"a\"]", false, "213 list order matters, then 231"),
        new("\"*\"", "\"IMAGE\"", true, "217 any received"),
        new("\"IMAGE\"", "\"*\"", true, "217 any input"),
        new("[\"a\"]", "\"*\"", true, "217 any input, list received"),
        new("\"COMFY_MATCHTYPE_V3\"", "\"IMAGE\"", true, "222 match type received"),
        new("\"LATENT\"", "\"COMFY_MATCHTYPE_V3\"", true, "222 match type input"),
        new("[\"a\",\"b\"]", "\"COMBO\"", true, "227 list into a V3 combo"),
        new("\"STRING\"", "\"COMBO\"", false, "227 only a list, then 247"),
        new("\"STRING\"", "[\"a\",\"b\"]", false, "231 into a V1 choice list"),
        new("[\"a\"]", "\"STRING\"", false, "231 list into a type"),
        new("null", "\"STRING\"", false, "231 not a string"),
        new("\"IMAGE,*\"", "\"LATENT\"", true, "239 any among received"),
        new("\"LATENT\"", "\" * ,MASK\"", true, "239 any among input, stripped"),
        new("\"STRING,BOOLEAN\"", "\"STRING,INT\"", true, "247 any overlap"),
        new("\"\u001cIMAGE\u001f\"", "\"IMAGE\"", true, "236 Python strip"),
        new("\"BOOLEAN\"", "\"STRING,INT\"", false, "247 no overlap"),
        new("\"IMAGE\"", "\"IMAGEX\"", false, "247 whole names only"),
    ];

    /// <summary>The non-strict cases of ComfyUI tests-unit/execution_test/validate_node_input_test.py (strict=True is never used by execution.py, and object_info cannot carry the __ne__ override cases).</summary>
    public static readonly (string Received, string Input, bool Accepted)[] UpstreamCases =
    [
        ("STRING", "STRING", true), ("STRING,INT", "STRING,INT", true), ("INT,STRING", "STRING,INT", true),
        ("STRING,BOOLEAN", "STRING,INT", true), ("STRING,INT", "INT,BOOLEAN", true), ("STRING", "STRING,INT", true),
        ("BOOLEAN", "STRING,INT", false), ("FLOAT", "STRING,INT", false), ("FLOAT,BOOLEAN", "STRING,INT", false),
        ("STRING, INT", "STRING,INT", true), ("STRING,INT", "STRING, INT", true), (" STRING , INT ", "STRING,INT", true), ("STRING,INT", " STRING , INT ", true),
        ("", "", true), ("STRING", "", false), ("", "STRING", false),
        ("STRING", "STRING,INT,BOOLEAN", true), ("STRING,INT,BOOLEAN", "STRING", true),
    ];

    /// <summary>validate_node_input is ported branch by branch, and agrees with ComfyUI's own unit test cases.</summary>
    [Test]
    public void TestValidateNodeInputBranches()
    {
        List<string> wrong = [];
        foreach (LinkTypeCase c in BranchCases)
        {
            if (ComfyInputValidation.ValidateNodeInput(JToken.Parse(c.Received), JToken.Parse(c.Input)) != c.Accepted)
            {
                wrong.Add($"{c.Branch}: {c.Received} into {c.Input} should be {(c.Accepted ? "accepted" : "rejected")}");
            }
        }
        foreach ((string received, string input, bool accepted) in UpstreamCases)
        {
            if (ComfyInputValidation.ValidateNodeInput(received, input) != accepted)
            {
                wrong.Add($"upstream: '{received}' into '{input}' should be {(accepted ? "accepted" : "rejected")}");
            }
        }
        Assert.That(wrong, Is.Empty, string.Join("\n", wrong));
    }

    /// <summary>Finalizes a node declaration (an object_info entry) for some prompt inputs, and returns the declared names per category.</summary>
    public static (string[] Required, string[] Optional) Declared(string nodeInfo, string liveInputs)
    {
        JObject finalized = ComfyInputValidation.FinalizedInputs(JObject.Parse(nodeInfo), JObject.Parse(liveInputs));
        static string[] names(JToken group) => group is JObject set ? [.. set.Properties().Select(p => p.Name).Order()] : [];
        return (names(finalized["required"]), names(finalized["optional"]));
    }

    /// <summary>A V3 object_info entry with the given "input" object.</summary>
    public static string V3(string input) => $$"""{ "input": {{input}}, "output": [], "output_matchtypes": null }""";

    /// <summary>A V1 node's inputs are its object_info "input" as is, "hidden" included; a V3 node drops "hidden" (_io.py:2032) and is told apart by its "output_matchtypes" key.</summary>
    [Test]
    public void TestV1AndV3Declarations()
    {
        const string inputs = """{ "required": { "a": ["INT", {}] }, "hidden": { "unique_id": "UNIQUE_ID" } }""";
        JObject v1 = ComfyInputValidation.FinalizedInputs(JObject.Parse($$"""{ "input": {{inputs}}, "output": [] }"""), []);
        Assert.That(v1["hidden"], Is.Not.Null);
        JObject v3 = ComfyInputValidation.FinalizedInputs(JObject.Parse(V3(inputs)), []);
        Assert.That(v3["hidden"], Is.Null);
        Assert.That(((JObject)v3["required"]).Properties().Select(p => p.Name), Is.EqualTo(new[] { "a" }));
    }

    /// <summary>Autogrow (_io.py:1179-1228) declares every possible name, required for the first "min" when the template input is required; the cases of dynamic_inputs_test.py and BatchImagesNode.</summary>
    [Test]
    public void TestAutogrowExpansion()
    {
        string rows = V3("""{ "required": { "rows": ["COMFY_AUTOGROW_V3", { "template": { "input": { "required": { "x": ["FLOAT", { "forceInput": true }] } }, "prefix": "item", "min": 0, "max": 10 } }] } }""");
        (string[] required, string[] optional) = Declared(rows, """{ "rows.item0": 0.5 }""");
        Assert.That(required, Is.Empty);
        Assert.That(optional, Is.EquivalentTo(Enumerable.Range(0, 10).Select(i => $"rows.item{i}")));
        Assert.That(Declared(rows, "{}").Optional, Has.Length.EqualTo(10), "declared whether or not the prompt sets any");
        string batch = WorkflowTraceTypingTests.DynamicObjectInfo;
        JObject batchInfo = (JObject)JObject.Parse(batch)["BatchImagesNode"];
        (required, optional) = Declared(batchInfo.ToString(), """{ "images.image0": ["1", 0], "images.image1": ["2", 0] }""");
        Assert.That(required, Is.EqualTo(new[] { "images.image0" }), "min=1 with a required template input");
        Assert.That(optional, Has.Length.EqualTo(49));
        string names = V3("""{ "optional": { "pick": ["COMFY_AUTOGROW_V3", { "template": { "input": { "required": {}, "optional": { "v": ["IMAGE", {}] } }, "names": ["front", "back"], "min": 2 } }] } }""");
        (required, optional) = Declared(names, "{}");
        Assert.That(required, Is.Empty, "an optional template input is never required");
        Assert.That(optional, Is.EquivalentTo(new[] { "pick.front", "pick.back" }));
    }

    /// <summary>DynamicCombo (_io.py:1266-1281) declares itself and its selected option's inputs only when the prompt sets it; the combo and nested-autogrow cases of dynamic_inputs_test.py.</summary>
    [Test]
    public void TestDynamicComboExpansion()
    {
        string combo = V3("""{ "required": { "rows": ["COMFY_DYNAMICCOMBO_V3", { "options": [{ "key": "on", "inputs": { "required": { "x": ["FLOAT", {}] } } }, { "key": "off", "inputs": { "required": {} } }] }] } }""");
        Assert.That(Declared(combo, """{ "rows": "on", "rows.x": 0.5 }""").Required, Is.EqualTo(new[] { "rows", "rows.x" }));
        Assert.That(Declared(combo, """{ "rows": "off" }""").Required, Is.EqualTo(new[] { "rows" }));
        Assert.That(Declared(combo, """{ "rows": "unknown" }""").Required, Is.Empty, "an unknown key declares nothing, not even the combo");
        Assert.That(Declared(combo, "{}").Required, Is.Empty, "unset, nothing is declared");
        string nested = V3("""{ "required": { "rows": ["COMFY_DYNAMICCOMBO_V3", { "options": [{ "key": "on", "inputs": { "required": { "items": ["COMFY_AUTOGROW_V3", { "template": { "input": { "required": { "x": ["FLOAT", {}] } }, "prefix": "item", "min": 0, "max": 10 } }] } } }] }] } }""");
        (string[] required, string[] optional) = Declared(nested, """{ "rows": "on", "rows.items.item0": 0.5 }""");
        Assert.That(required, Is.EqualTo(new[] { "rows" }));
        Assert.That(optional, Does.Contain("rows.items.item0").And.Length.EqualTo(10));
        JObject resize = (JObject)JObject.Parse(WorkflowTraceTypingTests.DynamicObjectInfo)["ResizeImageMaskNode"];
        Assert.That(Declared(resize.ToString(), """{ "input": ["1", 0], "resize_type": "scale to multiple", "resize_type.multiple": 8, "scale_method": "lanczos" }""").Required,
            Is.EqualTo(new[] { "input", "resize_type", "resize_type.multiple", "scale_method" }));
        Assert.That(Declared(resize.ToString(), """{ "input": ["1", 0], "resize_type": "match size", "resize_type.match": ["2", 0], "resize_type.crop": "center", "scale_method": "area" }""").Required,
            Is.EqualTo(new[] { "input", "resize_type", "resize_type.crop", "resize_type.match", "scale_method" }));
    }

    /// <summary>DynamicSlot (_io.py:1321-1328) declares itself and its inputs only when the prompt sets it.</summary>
    [Test]
    public void TestDynamicSlotExpansion()
    {
        string slot = V3("""{ "optional": { "mode": ["COMFY_DYNAMICSLOT_V3", { "slotType": "FLOAT", "inputs": { "required": { "extra": ["INT", {}] } } }] } }""");
        (string[] required, string[] optional) = Declared(slot, """{ "mode": 1.0, "mode.extra": 2 }""");
        Assert.That(required, Is.EqualTo(new[] { "mode.extra" }));
        Assert.That(optional, Is.EqualTo(new[] { "mode" }));
        (required, optional) = Declared(slot, "{}");
        Assert.That(required.Concat(optional), Is.Empty);
    }

    /// <summary>The DynamicGroup used by the cases below: io_dynamic_group_test.py test_every_submitted_row_keeps_template_requirements.</summary>
    public static string Group(int min, int max = 20) => V3($$"""{ "required": { "rows": ["COMFY_DYNAMICGROUP_V3", { "template": { "required": { "name": ["STRING", { "multiline": false }], "weight": ["FLOAT", { "default": 1.0 }] }, "optional": { "enabled": ["BOOLEAN", {}] } }, "min": {{min}}, "max": {{max}}, "group_name": "Group" }] } }""");

    /// <summary>DynamicGroup (_io.py:1407-1446) declares the fields of every submitted row and raises on malformed keys and row counts, with ComfyUI's messages; the cases of io_dynamic_group_test.py.</summary>
    [Test]
    public void TestDynamicGroupExpansion()
    {
        foreach (int minimum in new[] { 0, 1, 2 })
        {
            (string[] required, string[] optional) = Declared(Group(minimum), """{ "rows.0.name": "A", "rows.0.weight": 0.8, "rows.2.name": "C" }""");
            Assert.That(required, Is.EquivalentTo(new[] { "rows.0.name", "rows.0.weight", "rows.2.name", "rows.2.weight" }));
            Assert.That(optional, Is.EquivalentTo(new[] { "rows.0.enabled", "rows.2.enabled" }));
        }
        foreach (string key in new[] { "rows.foo.name", "rows.-1.name", "rows.01.name", "rows.+1.name", "rows.١.name", "rows.0", "rows..name", "rows.0.unknown", "rows.0.name.extra" })
        {
            ComfyInputValidation.DynamicInputError error = Assert.Throws<ComfyInputValidation.DynamicInputError>(() => Declared(Group(0), new JObject() { [key] = 1.0 }.ToString()), key);
            Assert.That(error.Message, Does.Contain(key));
            Assert.That(error.InputName, Is.EqualTo(key));
        }
        Assert.That(Assert.Throws<ComfyInputValidation.DynamicInputError>(() => Declared(Group(0, 1), """{ "rows.0.name": "A", "rows.2.name": "C" }""")).Message, Does.Contain("exceeds the index limit of 0"));
        Assert.That(Assert.Throws<ComfyInputValidation.DynamicInputError>(() => Declared(Group(0), """{ "rows.20.name": "A" }""")).Message, Does.Contain("exceeds the index limit of 19"));
        Assert.That(Assert.Throws<ComfyInputValidation.DynamicInputError>(() => Declared(Group(0), """{ "rows.99999999999999999999999.name": "A" }""")).Message, Does.Contain("exceeds the index limit of 19"), "an index past the long range");
        Assert.That(Assert.Throws<ComfyInputValidation.DynamicInputError>(() => Declared(Group(2), """{ "rows.2.name": "A" }""")).Message, Does.Contain("expected between 2 and"));
        (string[] noRequired, string[] noOptional) = Declared(Group(0), """{ "rows_summary.value": 1.0 }""");
        Assert.That(noRequired.Concat(noOptional), Is.Empty, "another dotted name is not a row");
    }

    /// <summary>A workflow with a source node of each class used by <see cref="TestCheckLinkBranches"/>.</summary>
    public const string LinkWorkflow = """
        {
            "1": { "class_type": "CheckpointLoaderSimple", "inputs": {} },
            "2": { "class_type": "SwarmInputDropdown", "inputs": {} },
            "3": { "class_type": "NotInObjectInfo", "inputs": {} },
            "4": { "inputs": {} }
        }
        """;

    /// <summary>object_info for <see cref="LinkWorkflow"/>: SwarmInputDropdown is SwarmComfyCommon/SwarmInputNodes.py lines 177-194, and SwarmInputLookalike has the same outputs without Swarm's mismatch exemption.</summary>
    public const string LinkObjectInfo = """
        {
            "CheckpointLoaderSimple": { "input": {}, "output": ["MODEL", "CLIP", "VAE"] },
            "SwarmInputDropdown": { "input": {}, "output": ["STRING", ""] },
            "SwarmInputLookalike": { "input": {}, "output": ["STRING", ""] }
        }
        """;

    /// <summary>The link branch of execution.py validate_inputs (lines 928-964): list shape, source lookup, Python indexing of the output slot, the type check, and Swarm's exemption for its dropdown inputs.</summary>
    [Test]
    public void TestCheckLinkBranches()
    {
        JObject workflow = JObject.Parse(LinkWorkflow), objectInfo = JObject.Parse(LinkObjectInfo);
        string check(string link, string inputType, out bool typeMismatch) => ComfyInputValidation.CheckLink(JArray.Parse(link), JToken.Parse(inputType), workflow, objectInfo, out typeMismatch);
        Assert.That(check("""["1", 2]""", "\"VAE\"", out bool mismatch), Is.Null);
        Assert.That(check("""["1", -1]""", "\"VAE\"", out _), Is.Null, "a negative slot counts from the end");
        Assert.That(check("""["1", true]""", "\"CLIP\"", out _), Is.Null, "a bool slot is 0 or 1");
        Assert.That(check("""["1", 1]""", "\"VAE\"", out mismatch), Is.EqualTo("expects VAE, gets CLIP from 1:1:CheckpointLoaderSimple"));
        Assert.That(mismatch, Is.True);
        Assert.That(check("""["1", 0, 1]""", "\"MODEL\"", out mismatch), Does.StartWith("is a list of 3 items"));
        Assert.That(mismatch, Is.False);
        Assert.That(check("""[1, 0]""", "\"MODEL\"", out _), Does.Contain("not a string"));
        Assert.That(check("""["9", 0]""", "\"MODEL\"", out _), Does.Contain("not in the workflow"));
        Assert.That(check("""["4", 0]""", "\"MODEL\"", out _), Does.Contain("no class_type"));
        Assert.That(check("""["3", 0]""", "\"MODEL\"", out _), Is.Null, "an unlisted source class is reported elsewhere");
        Assert.That(check("""["1", 3]""", "\"MODEL\"", out _), Is.EqualTo("links to output slot 3 of 1:CheckpointLoaderSimple, which has 3 outputs"));
        Assert.That(check("""["1", -4]""", "\"MODEL\"", out _), Does.Contain("which has 3 outputs"));
        Assert.That(check("""["1", 1.5]""", "\"MODEL\"", out _), Does.Contain("not an integer"));
        Assert.That(check("""["2", 1]""", """["euler", "ddim"]""", out _), Is.Null, "SwarmInternalUtil.py drops mismatches from SwarmInputDropdown");
        workflow["2"]["class_type"] = "SwarmInputLookalike";
        Assert.That(check("""["2", 1]""", """["euler", "ddim"]""", out mismatch), Is.EqualTo("expects [2 choices], gets \"\" from 2:1:SwarmInputLookalike"), "the same link from a class without the exemption");
        Assert.That(mismatch, Is.True);
    }
}
