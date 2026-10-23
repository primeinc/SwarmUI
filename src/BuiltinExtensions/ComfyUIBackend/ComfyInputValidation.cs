using System.Numerics;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace SwarmUI.Builtin_ComfyUIBackend;

/// <summary>Ports of the parts of ComfyUI's prompt validation that read the graph's structure, so a record can say which links and inputs ComfyUI rejects without running it.
/// Sources are ComfyUI 7ef81780: comfy_execution/validation.py, execution.py validate_inputs, and comfy_api/latest/_io.py. Node declarations come from object_info (server.py node_info, _io.py get_v1_info), which is INPUT_TYPES and RETURN_TYPES as ComfyUI validates them.
/// Value checks (number ranges, choice lists, custom VALIDATE_INPUTS) are left to ComfyUI.</summary>
public static class ComfyInputValidation
{
    /// <summary>The "any" type (_io.py:937 AnyType).</summary>
    public const string AnyType = "*";

    /// <summary>The V3 match type, which ComfyUI's frontend checks instead (_io.py:1017 MatchType).</summary>
    public const string MatchType = "COMFY_MATCHTYPE_V3";

    /// <summary>The V3 choice-list type (_io.py:344 Combo).</summary>
    public const string ComboType = "COMBO";

    /// <summary>Swarm's input nodes whose return type mismatches ComfyUI does not report: SwarmComfyCommon/SwarmInternalUtil.py lines 61-79 wrap execution.validate_inputs and drop return_type_mismatch errors linked from these classes.</summary>
    public static readonly string[] MismatchExemptSources = ["SwarmInputModelName", "SwarmInputDropdown"];

    /// <summary>Returns true for a character Python's str.isspace accepts, which is what str.strip removes: .NET's <see cref="char.IsWhiteSpace(char)"/> plus U+001C to U+001F, which Python counts (Unicode bidirectional classes B and S) and .NET does not.</summary>
    public static bool IsPythonWhitespace(char c)
    {
        return char.IsWhiteSpace(c) || (c >= '\u001C' && c <= '\u001F');
    }

    /// <summary>Python's str.strip() with no argument.</summary>
    public static string PythonStrip(string text)
    {
        int start = 0;
        int end = text.Length;
        while (start < end && IsPythonWhitespace(text[start]))
        {
            start++;
        }
        while (end > start && IsPythonWhitespace(text[end - 1]))
        {
            end--;
        }
        return text[start..end];
    }

    /// <summary>Python's == on two values decoded from JSON: lists compare item by item, dicts by keys and values, and numbers by value, with a bool equal to 0 or 1.</summary>
    public static bool PythonEquals(JToken a, JToken b)
    {
        if (a is JArray arrA && b is JArray arrB)
        {
            return arrA.Count == arrB.Count && arrA.Zip(arrB).All(pair => PythonEquals(pair.First, pair.Second));
        }
        if (a is JObject objA && b is JObject objB)
        {
            return objA.Count == objB.Count && objA.Properties().All(p => objB.TryGetValue(p.Name, out JToken other) && PythonEquals(p.Value, other));
        }
        if (a is not JValue valA || b is not JValue valB)
        {
            return false;
        }
        static bool isNumber(JValue v) => v.Type is JTokenType.Integer or JTokenType.Float or JTokenType.Boolean;
        static BigInteger asInteger(JValue v) => v.Type == JTokenType.Boolean ? (v.Value<bool>() ? 1 : 0) : IntegerOf(v);
        if (isNumber(valA) && isNumber(valB))
        {
            if (valA.Type != JTokenType.Float && valB.Type != JTokenType.Float)
            {
                return asInteger(valA) == asInteger(valB);
            }
            return valA.Value<double>() == valB.Value<double>();
        }
        if (valA.Type == JTokenType.String && valB.Type == JTokenType.String)
        {
            return (string)valA == (string)valB;
        }
        return valA.Type == JTokenType.Null && valB.Type == JTokenType.Null;
    }

    /// <summary>Returns the value of a JSON integer, which Newtonsoft holds as a long or, past the long range, a <see cref="BigInteger"/>.</summary>
    public static BigInteger IntegerOf(JValue value)
    {
        return value.Value is BigInteger big ? big : new BigInteger(value.Value<long>());
    }

    /// <summary>Returns the text of a JSON string value, or null if the token is not a string.</summary>
    public static string StringOf(JToken token)
    {
        return token is JValue { Type: JTokenType.String } value ? (string)value : null;
    }

    /// <summary>Port of validation.py validate_node_input (lines 193-247) with strict=False, the only way execution.py:950 calls it.
    /// Returns true if ComfyUI accepts a link from an output of type <paramref name="receivedType"/> into an input declared as <paramref name="inputType"/>; each is a type string, or a choice list for a V1 node.
    /// A legacy "any" type that is a str subclass overriding __ne__ (validation.py:211-214) reaches object_info as its plain string, so only that string is compared here.</summary>
    public static bool ValidateNodeInput(JToken receivedType, JToken inputType)
    {
        // validation.py:213
        if (PythonEquals(receivedType, inputType))
        {
            return true;
        }
        string received = StringOf(receivedType);
        string input = StringOf(inputType);
        // validation.py:217
        if (received == AnyType || input == AnyType)
        {
            return true;
        }
        // validation.py:222
        if (received == MatchType || input == MatchType)
        {
            return true;
        }
        // validation.py:227
        if (receivedType is JArray && input == ComboType)
        {
            return true;
        }
        // validation.py:231
        if (received is null || input is null)
        {
            return false;
        }
        // validation.py:235-247
        HashSet<string> receivedTypes = [.. received.Split(',').Select(PythonStrip)];
        HashSet<string> inputTypes = [.. input.Split(',').Select(PythonStrip)];
        if (receivedTypes.Contains(AnyType) || inputTypes.Contains(AnyType))
        {
            return true;
        }
        return receivedTypes.Overlaps(inputTypes);
    }

    /// <summary>Returns true if an object_info entry is a V3 node (comfy_api io.ComfyNode). Its entry is asdict(NodeInfoV1) (_io.py:1747-1771, 2270-2273), which always has an "output_matchtypes" key; server.py node_info (lines 844-887) never writes one for a V1 node.</summary>
    public static bool IsV3(JObject nodeInfo)
    {
        return nodeInfo.ContainsKey("output_matchtypes");
    }

    /// <summary>Thrown where _io.py raises DynamicInputError (_io.py:1330); execution.py:888-896 reports it as invalid_dynamic_input and rejects the node.</summary>
    public class DynamicInputError(string inputName, string message) : Exception(message)
    {
        /// <summary>The prompt input the error is about.</summary>
        public string InputName = inputName;
    }

    /// <summary>Returns a node's inputs as execution.py validate_inputs sees them (lines 883-909) for one prompt node: a "required" and an "optional" map from input name to declaration.
    /// For a V3 node this is _io.py get_finalized_class_inputs (lines 2022-2046) with the node's prompt inputs, which drops "hidden" and expands dynamic inputs to the prefixed names the prompt uses; a V1 node's object_info "input" is returned as is, "hidden" included.
    /// Throws <see cref="DynamicInputError"/> where ComfyUI does, and other exceptions where ComfyUI's expansion would fail on a malformed declaration.</summary>
    public static JObject FinalizedInputs(JObject nodeInfo, JObject liveInputs)
    {
        JObject classInputs = nodeInfo["input"] as JObject ?? [];
        if (!IsV3(nodeInfo))
        {
            return classInputs;
        }
        JObject outDict = new() { ["required"] = new JObject(), ["optional"] = new JObject() };
        JObject withoutHidden = [];
        foreach (JProperty category in classInputs.Properties())
        {
            if (category.Name != "hidden")
            {
                withoutHidden[category.Name] = category.Value;
            }
        }
        ParseClassInputs(outDict, liveInputs, withoutHidden, null);
        return outDict;
    }

    /// <summary>Port of _io.py handle_prefix (lines 1078-1083).</summary>
    public static List<string> HandlePrefix(List<string> prefixList, string id)
    {
        List<string> result = prefixList is null ? [] : [.. prefixList];
        if (id is not null)
        {
            result.Add(id);
        }
        return result;
    }

    /// <summary>Port of _io.py finalize_prefix (lines 1085-1091).</summary>
    public static string FinalizePrefix(List<string> prefixList, string id = null)
    {
        if (prefixList is null)
        {
            return id;
        }
        return id is null ? string.Join('.', prefixList) : string.Join('.', [.. prefixList, id]);
    }

    /// <summary>Port of _io.py parse_class_inputs (lines 2048-2062): copies plain inputs under their prefixed names and expands dynamic ones (_io.py:1655-1663 setup_dynamic_input_funcs).</summary>
    public static void ParseClassInputs(JObject outDict, JObject liveInputs, JObject currDict, List<string> currPrefix)
    {
        foreach (JProperty category in currDict.Properties())
        {
            foreach (JProperty input in ((JObject)category.Value).Properties())
            {
                JArray value = (JArray)input.Value;
                switch (StringOf(value[0]))
                {
                    case "COMFY_AUTOGROW_V3":
                        ExpandAutogrow(outDict, liveInputs, value, HandlePrefix(currPrefix, input.Name));
                        break;
                    case "COMFY_DYNAMICCOMBO_V3":
                        ExpandDynamicCombo(outDict, liveInputs, value, category.Name, HandlePrefix(currPrefix, input.Name));
                        break;
                    case "COMFY_DYNAMICSLOT_V3":
                        ExpandDynamicSlot(outDict, liveInputs, value, category.Name, HandlePrefix(currPrefix, input.Name));
                        break;
                    case "COMFY_DYNAMICGROUP_V3":
                        ExpandDynamicGroup(outDict, liveInputs, value, HandlePrefix(currPrefix, input.Name));
                        break;
                    default:
                        ((JObject)outDict[category.Name])[FinalizePrefix(currPrefix, input.Name)] = value;
                        break;
                }
            }
        }
    }

    /// <summary>Port of _io.py Autogrow._expand_schema_for_dynamic (lines 1179-1228): every possible name is declared, required for the first "min" names when the template input is required, optional otherwise.</summary>
    public static void ExpandAutogrow(JObject outDict, JObject liveInputs, JArray value, List<string> currPrefix)
    {
        JObject template = (JObject)value[1]["template"];
        JObject input = (JObject)template["input"];
        long min;
        List<string> names;
        if (template.ContainsKey("names"))
        {
            min = template["min"].Value<long>();
            names = [.. ((JArray)template["names"]).Select(n => (string)n)];
        }
        else if (template.ContainsKey("prefix"))
        {
            string prefix = (string)template["prefix"];
            min = template["min"].Value<long>();
            long max = template["max"].Value<long>();
            names = [];
            for (long i = 0; i < max; i++)
            {
                names.Add($"{prefix}{i}");
            }
        }
        else
        {
            throw new InvalidOperationException("Autogrow template has neither names nor prefix");
        }
        JToken templateInput = null;
        bool templateRequired = true;
        foreach (JProperty category in input.Properties())
        {
            JObject dictInput = (JObject)category.Value;
            if (dictInput.Count == 0)
            {
                continue;
            }
            templateInput = dictInput.Properties().First().Value;
            templateRequired = category.Name == "required";
            break;
        }
        if (templateInput is null)
        {
            throw new InvalidOperationException("template_input could not be determined from required or optional; this should never happen.");
        }
        JObject newDict = [];
        for (int i = 0; i < names.Count; i++)
        {
            string expectedId = FinalizePrefix(currPrefix, names[i]);
            string category = i < min && templateRequired ? "required" : "optional";
            ((JObject)outDict[category])[expectedId] = templateInput;
            JObject typeDict = newDict[category] as JObject ?? (JObject)(newDict[category] = new JObject());
            if (liveInputs.ContainsKey(expectedId))
            {
                typeDict[names[i]] = templateInput;
            }
        }
        ParseClassInputs(outDict, liveInputs, newDict, currPrefix);
    }

    /// <summary>Port of _io.py DynamicCombo._expand_schema_for_dynamic (lines 1266-1281): only when the prompt sets the combo, its selected option's inputs are declared, and so is the combo itself.</summary>
    public static void ExpandDynamicCombo(JObject outDict, JObject liveInputs, JArray value, string inputType, List<string> currPrefix)
    {
        string finalizedId = FinalizePrefix(currPrefix);
        if (!liveInputs.TryGetValue(finalizedId, out JToken key))
        {
            return;
        }
        JObject selectedOption = ((JArray)value[1]["options"]).OfType<JObject>().FirstOrDefault(option => PythonEquals(option["key"], key));
        if (selectedOption is not null)
        {
            ParseClassInputs(outDict, liveInputs, (JObject)selectedOption["inputs"], currPrefix);
            ((JObject)outDict[inputType])[finalizedId] = value;
        }
    }

    /// <summary>Port of _io.py DynamicSlot._expand_schema_for_dynamic (lines 1321-1328): only when the prompt sets the slot, its inputs are declared, and so is the slot itself.</summary>
    public static void ExpandDynamicSlot(JObject outDict, JObject liveInputs, JArray value, string inputType, List<string> currPrefix)
    {
        string finalizedId = FinalizePrefix(currPrefix);
        if (liveInputs.ContainsKey(finalizedId))
        {
            ParseClassInputs(outDict, liveInputs, (JObject)value[1]["inputs"], currPrefix);
            ((JObject)outDict[inputType])[finalizedId] = value;
        }
    }

    /// <summary>Port of _io.py _DynamicGroup._expand_schema_for_dynamic (lines 1407-1446): the fields of every submitted row are declared under "group.row.field", with the row-key and row-count checks that raise <see cref="DynamicInputError"/>.</summary>
    public static void ExpandDynamicGroup(JObject outDict, JObject liveInputs, JArray value, List<string> currPrefix)
    {
        JObject info = (JObject)value[1];
        long minRows = info["min"]?.Value<long>() ?? 0;
        long maxRows = info["max"]?.Value<long>() ?? 20;
        JObject template = info["template"] as JObject ?? [];
        Dictionary<string, (JToken Value, string Category)> fieldSpecs = [];
        foreach (string category in new[] { "required", "optional" })
        {
            if (template[category] is JObject fields)
            {
                foreach (JProperty field in fields.Properties())
                {
                    fieldSpecs[field.Name] = (field.Value, category);
                }
            }
        }
        string finalizedPrefix = FinalizePrefix(currPrefix);
        string prefix = finalizedPrefix + ".";
        HashSet<BigInteger> presentRows = [];
        foreach (JProperty live in liveInputs.Properties())
        {
            string key = live.Name;
            if (!key.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }
            string rest = key[prefix.Length..];
            int dot = rest.IndexOf('.');
            string index = dot < 0 ? rest : rest[..dot];
            string fieldId = dot < 0 ? "" : rest[(dot + 1)..];
            if (dot < 0 || index.Length == 0 || !index.All(c => c >= '0' && c <= '9') || (index.Length > 1 && index[0] == '0') || !fieldSpecs.ContainsKey(fieldId))
            {
                throw new DynamicInputError(key, $"Invalid DynamicGroup input key '{key}'; expected '{finalizedPrefix}.<index>.<template field>'.");
            }
            BigInteger row = BigInteger.Parse(index);
            if (row >= maxRows)
            {
                throw new DynamicInputError(key, $"DynamicGroup input '{key}' exceeds the index limit of {maxRows - 1} (max={maxRows}).");
            }
            presentRows.Add(row);
        }
        if (presentRows.Count < minRows || presentRows.Count > maxRows)
        {
            throw new DynamicInputError(finalizedPrefix, $"DynamicGroup input '{finalizedPrefix}' received {presentRows.Count} rows; expected between {minRows} and {maxRows}.");
        }
        BigInteger lastRow = presentRows.Count == 0 ? -1 : presentRows.Max();
        for (BigInteger row = 0; row <= lastRow; row++)
        {
            if (!presentRows.Contains(row))
            {
                continue;
            }
            foreach ((string fieldId, (JToken fieldValue, string category)) in fieldSpecs)
            {
                ((JObject)outDict[category])[$"{finalizedPrefix}.{row}.{fieldId}"] = fieldValue;
            }
        }
    }

    /// <summary>Returns the type text of a declaration or return type for a record: a type string as is (quoted when empty), a choice list as its length.</summary>
    public static string TypeText(JToken type)
    {
        return type switch
        {
            JArray list => $"[{list.Count} choices]",
            JValue { Type: JTokenType.String } text => (string)text == "" ? "\"\"" : (string)text,
            _ => type?.ToString(Formatting.None) ?? "null"
        };
    }

    /// <summary>Port of the link branch of execution.py validate_inputs (lines 928-964) for one declared input whose prompt value is a list, which ComfyUI always reads as a link.
    /// Returns null if ComfyUI accepts the link, or if <paramref name="objectInfo"/> does not list the source's class (ComfyUI rejects the prompt for that class first, execution.py:1165-1180); otherwise the reason ComfyUI rejects it, with <paramref name="typeMismatch"/> set when the reason is return_type_mismatch.
    /// Swarm's validate_inputs wrapper drops mismatches from <see cref="MismatchExemptSources"/>. object_info does not say which nodes' VALIDATE_INPUTS take input_types, which skips the type check (execution.py:950), so such a node's mismatches are still reported.</summary>
    public static string CheckLink(JArray link, JToken inputType, JObject prompt, JObject objectInfo, out bool typeMismatch)
    {
        typeMismatch = false;
        // execution.py:931-943
        if (link.Count != 2)
        {
            return $"is a list of {link.Count} items, which ComfyUI reads as a link and rejects: a link is [node_id, slot]";
        }
        // execution.py:945-947
        string sourceId = StringOf(link[0]);
        if (sourceId is null)
        {
            return $"links to node {link[0].ToString(Formatting.None)}, which is not a string, so ComfyUI does not find it";
        }
        if (prompt[sourceId] is not JObject source)
        {
            return $"links to node {sourceId}, which is not in the workflow";
        }
        string sourceClass = StringOf(source["class_type"]);
        if (sourceClass is null)
        {
            return $"links to node {sourceId}, which has no class_type";
        }
        if (objectInfo[sourceClass] is not JObject sourceInfo)
        {
            return null;
        }
        JArray returnTypes = sourceInfo["output"] as JArray ?? [];
        // execution.py:948, with Python list indexing: an int or bool index, negative from the end
        JToken slot = link[1];
        BigInteger index;
        if (slot.Type == JTokenType.Integer)
        {
            index = IntegerOf((JValue)slot);
        }
        else if (slot.Type == JTokenType.Boolean)
        {
            index = slot.Value<bool>() ? 1 : 0;
        }
        else
        {
            return $"links to output slot {slot.ToString(Formatting.None)} of {sourceId}:{sourceClass}, which is not an integer";
        }
        if (index < 0)
        {
            index += returnTypes.Count;
        }
        if (index < 0 || index >= returnTypes.Count)
        {
            return $"links to output slot {slot} of {sourceId}:{sourceClass}, which has {returnTypes.Count} outputs";
        }
        JToken receivedType = returnTypes[(int)index];
        // execution.py:950
        if (ValidateNodeInput(receivedType, inputType) || MismatchExemptSources.Contains(sourceClass))
        {
            return null;
        }
        typeMismatch = true;
        return $"expects {TypeText(inputType)}, gets {TypeText(receivedType)} from {sourceId}:{slot}:{sourceClass}";
    }
}
