"""Run real mouse/touch interaction checks against the local Streamlit server."""
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright, expect

OUT = Path(__file__).resolve().parents[1] / 'results' / 'app_browser'
OUT.mkdir(parents=True, exist_ok=True)
URL = 'http://localhost:8501'


def draw_one(page):
    box = page.get_by_label('Draw one digit from zero to nine', exact=True).bounding_box()
    x, y = box['x'] + box['width'] * .5, box['y'] + box['height'] * .22
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x, box['y'] + box['height'] * .8, steps=15)
    page.mouse.up()
    expect(page.get_by_role('button', name='Predict')).to_be_enabled()


def wait_digit(page):
    expect(page.locator('.prediction-digit')).to_have_text(__import__('re').compile(r'^[0-9]$'), timeout=60000)
    return page.locator('.prediction-digit').inner_text()


with sync_playwright() as pw:
    browser = pw.chromium.launch(channel='msedge', headless=True)
    context = browser.new_context(viewport={'width': 1280, 'height': 1000})
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(URL, wait_until='domcontentloaded')
    expect(page.get_by_role('heading', name='Draw a digit.')).to_be_visible(timeout=120000)
    expect(page.get_by_role('button', name='Predict')).to_be_enabled(timeout=120000)
    page.get_by_role('button', name='Predict').click()
    expect(page.get_by_text('The canvas is blank.', exact=False)).to_be_visible(timeout=30000)
    draw_one(page)
    page.get_by_role('button', name='Predict').click()
    results = [{'model': 'CNN', 'digit': wait_digit(page)}]
    assert results[0]['digit'] == '1', results
    expect(page.get_by_text('The canvas is blank.', exact=False)).not_to_be_visible(timeout=30000)
    expect(page.locator('[data-testid="stImage"] img')).to_be_visible(timeout=30000)
    page.wait_for_function("Array.from(document.querySelectorAll('[data-testid=stImage] img')).every(i=>i.complete&&i.naturalWidth>0)")
    expect(page.get_by_text('What the model sees')).to_be_visible()
    page.screenshot(path=str(OUT / 'desktop.png'), full_page=True)
    for name in ['MLP', 'Random Forest']:
        page.get_by_role('combobox').click()
        page.get_by_role('option', name=name, exact=True).click()
        expect(page.locator('.prediction-digit')).to_have_count(0)
        page.get_by_role('button', name='Predict').click()
        results.append({'model': name, 'digit': wait_digit(page)})
    # A new stroke must invalidate the old result; Clear resets both drawing/result.
    draw_one(page)
    expect(page.locator('.prediction-digit')).to_have_count(0)
    page.get_by_role('button', name='Clear', exact=True).click()
    expect(page.locator('.prediction-digit')).to_have_count(0)
    page.get_by_role('button', name='Predict').click()
    expect(page.get_by_text('The canvas is blank.', exact=False)).to_be_visible()
    page.get_by_text('Compare the three models', exact=True).click()
    expect(page.locator('[data-testid="stDataFrame"]')).to_be_visible()
    assert not page.locator('[data-testid="stException"]').count()
    mobile_context = browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
    mobile = mobile_context.new_page()
    mobile.on('pageerror', lambda error: errors.append(str(error)))
    mobile.goto(URL, wait_until='domcontentloaded')
    expect(mobile.get_by_role('button', name='Predict')).to_be_enabled(timeout=120000)
    canvas = mobile.get_by_label('Draw one digit from zero to nine', exact=True)
    canvas.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    cdp = mobile_context.new_cdp_session(mobile)
    x = box['x'] + box['width'] * .5
    y = box['y'] + box['height'] * .2
    cdp.send('Input.dispatchTouchEvent', {'type':'touchStart', 'touchPoints':[{'x':x,'y':y}]})
    for step in range(1, 12):
        cdp.send('Input.dispatchTouchEvent', {'type':'touchMove', 'touchPoints':[{'x':x,'y':y+box['height']*.6*step/11}]})
    cdp.send('Input.dispatchTouchEvent', {'type':'touchEnd', 'touchPoints':[]})
    mobile.get_by_role('button', name='Predict').click()
    mobile_digit = wait_digit(mobile)
    assert mobile_digit == '1', mobile_digit
    mobile.screenshot(path=str(OUT / 'mobile_drawing.png'), full_page=True)
    mobile.locator('.prediction-digit').scroll_into_view_if_needed()
    expect(mobile.locator('.prediction-digit')).to_be_visible()
    mobile.screenshot(path=str(OUT / 'mobile.png'), full_page=True)
    assert not errors, errors
    evidence = {'url':URL, 'status':'passed', 'desktop_predictions':results, 'touch_prediction':mobile_digit,
                'checks':['blank rejection','mouse drawing','three model choices','drawing preserved on model change',
                          'stale result invalidation','clear/reset','comparison table','touch drawing','no browser page errors'],
                'tested_unix':time.time()}
    (OUT / 'checks.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2))
    browser.close()
