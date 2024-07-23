const tapContainer = document.querySelector('.about');
const flex_Container = document.querySelectorAll('.contents_container');
const taps = document.querySelectorAll('.about > span');

function openCity(e){
    let elem = e.target;
    
    for (var i = 0; i < flex_Container.length; i++) {
        flex_Container[i].classList.remove('active');
        taps[i].classList.remove('on');
    }
    
    if(elem.matches('.nick_name')){
        flex_Container[0].classList.add('active');
        taps[0].classList.add('on');
    } else if(elem.matches('.like_mark')){
        flex_Container[1].classList.add('active');
        taps[1].classList.add('on');
    } else if(elem.matches('.book_mark')){
        flex_Container[2].classList.add('active');
        taps[2].classList.add('on');
    }
}

tapContainer.addEventListener('click', openCity);


document.getElementById('follow-list').addEventListener('click', function() {
    const userId = this.getAttribute('data-user-id');
    fetch(`/profile/${userId}/following-popup/`)
        .then(response => response.json())
        .then(data => {
            const popup = window.open("", "Following List", "width=400,height=600");
            popup.document.write("<div class='scroll_inner'>");
            data.forEach(user => {
                popup.document.write(`<div class='user'><div class='id'>${user.nickname}</div></div>`);
            });
            popup.document.write("</div>");
        })
        .catch(error => console.error('Error loading the following list:', error));
});