// Real Chromium DOM + production LoRA JavaScript, no mock Document/Option/Select.
import assert from 'node:assert/strict';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const scriptPath = path.join(root, 'src/wwwroot/js/genpage/gentab/loras.js');

async function withRealBrowser(fn) {
    const browser = await chromium.launch({ headless: true });
    try {
        const page = await browser.newPage();
        await page.setContent('<label for="input_prompt">Prompt</label>'
            + '<textarea id="input_prompt">scene &lt;segment:face,1,0.8&gt; portrait &lt;segment:hair,1,0.8&gt; hair</textarea>'
            + '<label for="scope">LoRA confinement</label><select id="scope"></select>');
        await page.addScriptTag({ path: scriptPath });
        await fn(page);
    } finally {
        await browser.close();
    }
}

test('native Chromium selector offers numbered segment choices', async () => {
    await withRealBrowser(async page => {
        const data = await page.evaluate(() => {
            const selector = document.querySelector('#scope');
            loraHelper.populateConfinementSelect(selector, '11', document.querySelector('#input_prompt').value);
            return { value: selector.value,
                options: Array.from(selector.options, o => ({ value: o.value, label: o.label })) };
        });
        assert.equal(data.value, '11');
        assert.match(data.options.find(o => o.value === '11').label, /Segment 1.*face/);
        assert.match(data.options.find(o => o.value === '12').label, /Segment 2.*hair/);
        assert.equal(data.options.find(o => o.value === '0').label, '@ Global');
    });
});

test('native selection persists when prompt no longer mentions that section', async () => {
    await withRealBrowser(async page => {
        await page.evaluate(() => loraHelper.populateConfinementSelect(
            document.querySelector('#scope'), '11'));
        await page.locator('#input_prompt').fill('new prompt without segments');
        const data = await page.evaluate(() => {
            const select = document.querySelector('#scope');
            loraHelper.populateConfinementSelect(select, select.value);
            return { value: select.value, label: select.selectedOptions[0]?.label };
        });
        assert.equal(data.value, '11');
        assert.match(data.label, /not in current prompt/);
    });
});

test('native selector matches server mixed region/segment numbering', async () => {
    await withRealBrowser(async page => {
        await page.locator('#input_prompt').fill(
            'scene <region:0,0,0.5,0.5> first <object:0,0,1,1> second <segment:face> portrait');
        const values = await page.evaluate(() => {
            const select = document.querySelector('#scope');
            loraHelper.populateConfinementSelect(select, '13');
            return Array.from(select.options, o => ({ value: o.value, text: o.text }));
        });
        assert.match(values.find(x => x.value === '11').text, /Region/);
        assert.match(values.find(x => x.value === '12').text, /Object/);
        assert.match(values.find(x => x.value === '13').text, /Segment/);
    });
});
