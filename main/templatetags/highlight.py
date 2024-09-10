import re
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(name='highlight')
def highlight(text, search):
    if search:
        # 정확히 검색어만 하이라이트되도록 수정
        pattern = re.compile(r'(\b{}\b)'.format(re.escape(search)), re.IGNORECASE)
        highlighted = pattern.sub(r'<span class="highlight">\1</span>', text)
        return mark_safe(highlighted)
    return text
