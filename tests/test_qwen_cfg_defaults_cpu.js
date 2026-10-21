// CPU/Node proof of the shipped model-switch CFG helper, without a browser
// or image model. Verify the real source function rather than a reimplementation.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const models = fs.readFileSync(
    path.join(__dirname, '..', 'src', 'wwwroot', 'js', 'genpage', 'gentab', 'models.js'), 'utf8');
const start = models.indexOf('function recommendedCfgOnModelSwitch(');
assert.notEqual(start, -1, 'Shipped model-switch helper must exist');
const end = models.indexOf('\n}\n', start);
assert.notEqual(end, -1, 'Shipped helper definition must be closed');
const body = models.slice(start, end + 2);
const select = vm.runInNewContext(body + '\nrecommendedCfgOnModelSwitch;');
const cases = [
    // Prior model, new model, current field, saved cookie, last auto value, expected.
    [null, 'qwen-image-2.1', '7', null, null, '1'],
    ['qwen-image-2.1', 'sdxl', '1', null, '1', '7'],
    ['sdxl', 'qwen-image-2.1', '5', null, null, null],
    ['sdxl', 'qwen-image-2.1', '7', '7', null, null],
    ['sdxl', 'qwen-image-2.1', '7', '1', null, null],
    ['sdxl', 'qwen-image-2.1', '7', '7', '7', '1'],
    ['sdxl', 'sd3', '7', null, null, null],
    ['qwen-image-2.1', 'sdxl', '1', '1', '1', '7'],
];
for (const args of cases) {
    const expected = args[5];
    const actual = select(...args.slice(0, 5));
    assert.equal(actual, expected, 'Wrong CFG after model switch ' + JSON.stringify(args));
}
// Guard the browser wiring: the recommended value must be *visible* and
// forwarded as the real CFG input, not hidden server-side prompt rewriting.
assert.match(models, /let previousCompat = this\.curCompatClass/);
assert.match(models, /recommendedCfgOnModelSwitch\(/);
assert.match(models, /setDirectParamValue\(cfgParam, recommended, cfgInput, false, false\)/);
console.log('Qwen native UI CFG model-switch controls: 8 passed');
