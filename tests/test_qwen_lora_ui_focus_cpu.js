// Exercise the actual shipped LoRA helper in a controlled DOM focus simulation.
// Proves which prompt pane supplies confinement IDs after clicking the select.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const file = fs.readFileSync(
    path.join(__dirname, '..', 'src', 'wwwroot', 'js', 'genpage', 'gentab', 'loras.js'),
    'utf8');
const start = file.indexOf('class LoraHelper {');
const end = file.lastIndexOf('loraHelper = new LoraHelper();');
assert.ok(start >= 0 && end > start);
const onlyHelper = file.slice(start, end);
const editors = {
    input_prompt: { id: 'input_prompt',
        value: '<segment:dog,0.8,0.1> replace with cat' },
    alt_prompt_textbox: { id: 'alt_prompt_textbox',
        value: '<segment:girl,0.8,0.1> replace with boy' },
};
const listener = {};
const doc = {
    activeElement: null,
    addEventListener(name, callback) { listener[name] = callback; },
    getElementById(name) { return editors[name] ?? null; },
    focus(item) {
        this.activeElement = item;
        listener.focusin?.({ target: item });
    },
};
class FakeOption {
    constructor(text, value) {
        this.text = text;
        this.value = value;
    }
}
const Helper = vm.runInNewContext(onlyHelper + '\nLoraHelper;', {
    document: doc, Option: FakeOption,
});
const helper = new Helper();
const select = {
    id: 'lora-confinement',
    options: [],
    value: '0',
    replaceChildren() { this.options = []; },
    add(option) { this.options.push(option); },
};
doc.focus(editors.alt_prompt_textbox);
assert.match(helper.getActiveConfinementPrompt(), /segment:girl/);
doc.focus(select);
assert.match(helper.getActiveConfinementPrompt(), /segment:girl/,
    'Clicking confinement select must not switch to the main prompt');
helper.populateConfinementSelect(select, '11');
assert.ok(select.options.some(option => option.value === '11'
    && option.text.includes('girl')));
assert.equal(select.value, '11');

doc.focus(editors.input_prompt);
doc.focus(select);
assert.match(helper.getActiveConfinementPrompt(), /segment:dog/,
    'Explicitly editing the main prompt must update remembered pane');
helper.populateConfinementSelect(select, '11');
assert.ok(select.options.some(option => option.value === '11'
    && option.text.includes('dog')));

const background = helper.getConfinementOptions(
    '<region:background> scenery <segment:girl,0.8,0.1> replacement', '11');
assert.equal(background.find(option => option.value === '11').label,
    'Segment 1: girl');
assert.equal(background.filter(option => /Background/.test(option.label)).length, 0);
assert.ok(!background.some(option => option.value === '12'),
    'Unsampled background region must not allocate a section');
const reset = helper.getConfinementOptions(
    '<segment:girl,0.8,0.1> A <region:end> B <segment:dog,0.8,0.1> C', '12');
assert.equal(reset.find(option => option.value === '11').label,
    'Segment 1: girl');
assert.equal(reset.find(option => option.value === '12').label,
    'Segment 2: dog');
delete editors.alt_prompt_textbox;
doc.activeElement = select;
helper.lastConfinementPromptId = 'alt_prompt_textbox';
assert.match(helper.getActiveConfinementPrompt(), /segment:dog/,
    'Removed alternate editor must safely fall back to the main prompt');
console.log('LoRA confinement prompt-focus and sampled section controls passed');
