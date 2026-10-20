'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = fs.readFileSync(path.join(__dirname, '..', 'src', 'wwwroot', 'js', 'genpage', 'gentab', 'loras.js'), 'utf8');

class MockOption {
    constructor(text, value) {
        this.text = text;
        this.value = String(value);
    }
}
class MockSelect {
    options = [];
    _value = '';
    get value() {
        return this._value;
    }
    set value(next) {
        const requested = String(next);
        // Like an actual HTML select, assigning a value absent from its options clears the selection.
        this._value = this.options.some(option => option.value === requested) ? requested : '';
    }
    add(option) {
        this.options.push(option);
    }
    replaceChildren() {
        this.options = [];
    }
}
const sandbox = { Option: MockOption };
vm.runInNewContext(source + '\nthis.TestLoraHelper = LoraHelper;', sandbox, { filename: 'loras.js' });
const helper = new sandbox.TestLoraHelper();

function optionsFor(prompt, selected = '0') {
    return new Map(Array.from(helper.getConfinementOptions(prompt, selected), item => [item.value, item.label]));
}

test('real prompt segment IDs 11/12 become selectable for every LoRA', () => {
    const opts = optionsFor('scene <segment:face,1,0.8> face <segment:body,0.6,0.5> body');
    assert.match(opts.get('11'), /Segment 1.*face/);
    assert.match(opts.get('12'), /Segment 2.*body/);
    assert.equal(opts.get('0'), '@ Global');
});

test('UI numbering counts preceding region/object and refiner reset like server autoConfine', () => {
    let opts = optionsFor('scene <region:0,0,0.5,0.5> stuff <object:0,0,1,1> else <segment:face> edit');
    assert.match(opts.get('11'), /Region/);
    assert.match(opts.get('12'), /Object/);
    assert.match(opts.get('13'), /Segment/);
    opts = optionsFor('scene <segment:hair> first <refiner> section <segment:face> second');
    assert.match(opts.get('11'), /Segment/);
});

test('existing segmented confinement never silently renders as Global', () => {
    const select = new MockSelect();
    helper.populateConfinementSelect(select, '11', '<segment:face> portrait');
    assert.equal(select.value, '11');
    assert.match(select.options.find(item => item.value === '11').text, /Segment/);
    helper.populateConfinementSelect(select, '19', '<segment:face> portrait');
    assert.equal(select.value, '19');
    assert.match(select.options.find(item => item.value === '19').text, /not in current prompt/);
});

test('no segment prompt leaves stage choices and preserves other stage confinements', () => {
    const options = optionsFor('portrait', '1');
    assert.equal(options.has('11'), false);
    assert.equal(options.get('1'), 'Refiner');
    assert.equal(options.get('5'), 'Base');
    assert.equal(options.get('0'), '@ Global');
});


test('existing browser LoRA row renders selected segment and refreshes without changing its value', () => {
    const primary = { value: 'scene <segment:face> identity edit' };
    sandbox.document = {
        activeElement: primary,
        getElementById(id) {
            return id === 'input_prompt' ? primary : id === 'input_loras' ? {} : null;
        }
    };
    sandbox.getRequiredElementById = () => ({ style: {}, innerText: '' });
    sandbox.genTabLayout = { reapplyPositions() {} };
    sandbox.setTimeout = callback => { callback(); return 1; };
    const select = new MockSelect();
    helper.selected = [{ name: 'face', weight: 0.8, confinement: 11 }];
    helper.rendered = {
        face: {
            weightInput: { value: 1 },
            confinementInput: select,
            fixSize() {}
        }
    };
    helper.rebuildUI();
    assert.equal(select.value, '11');
    assert.match(select.options.find(option => option.value === '11').text, /Segment 1.*face/);
    assert.equal(helper.rendered.face.weightInput.value, 0.8);

    // Editing the prompt must not silently reassign an existing adapter to Global.
    primary.value = 'scene without segmentation';
    helper.refreshConfinementOptions();
    assert.equal(select.value, '11');
    assert.match(select.options.find(option => option.value === '11').text, /not in current prompt/);
});


test('segment selection reaches hidden LoRA section confinement request parameter', () => {
    const loraInput = {
        selectedOptions: [{ value: 'face' }],
        options: [{ value: 'face' }]
    };
    const weightsInput = { value: '0.8' };
    const confinementInput = { value: '' };
    const prompt = { value: '<segment:face> portrait' };
    const elements = {
        input_loras: loraInput,
        input_loraweights: weightsInput,
        input_lorasectionconfinement: confinementInput,
        input_prompt: prompt
    };
    sandbox.document = {
        activeElement: prompt,
        getElementById: id => elements[id] ?? null
    };
    sandbox.arraysEqual = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    sandbox.triggerChangeFor = () => {};
    helper.selected = [{ name: 'face', weight: 0.8, confinement: '11' }];
    helper.rebuildParams();
    assert.equal(confinementInput.value, '11',
        'The selected section must be serialized into the parameter sent by getGenInput.');
    helper.selected[0].confinement = '0';
    helper.rebuildParams();
    assert.equal(confinementInput.value, '',
        'Only globally scoped adapters should use the legacy empty confinement field.');
});
